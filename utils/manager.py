import logging
from copy import deepcopy
import torch
import torch.nn as nn
from tqdm import tqdm
from . import Metric, classification_accuracy
from .prune import SparsePruner
import models.layers as nl


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, shared_layer_info, masks, train_loader, val_loader):
        self.args = args
        self.model = model
        self.model.module.masks = masks
        self.shared_layer_info = shared_layer_info
        self.inference_dataset_idx = self.model.module.datasets.index(args.dataset) + 1
        self.pruner = SparsePruner(self.model, masks, self.args, self.inference_dataset_idx)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.sample = self.args.sample
        return

    def train(self, optimizers, schedulers, epoch_idx, curr_lrs):
        # Set model to training mode
        self.sample = self.args.sample
        self.model.train()

        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')
        with tqdm(total=len(self.train_loader),
                  desc='Train Ep. #{}: '.format(epoch_idx + 1),
                  disable=False,
                  ascii=True) as t:
            for info in self.train_loader:
                if self.args.cuda:
                    input_ids = info['input_ids'].cuda()
                    attention_mask = info['attention_mask'].cuda()
                    target = info['label'].cuda()
                curr_lrs[0] = optimizers[0].param_groups[0]['lr']
                curr_lrs[1] = optimizers[1].param_groups[0]['lr']
                optimizers.zero_grad()
                # Do forward-backward.
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=target, sample=self.sample)

                num = input_ids.size(0)
                train_accuracy.update(classification_accuracy(output[1], target), num)

                loss = output[0].mean()
                train_loss.update(loss, num)
                loss.backward()

                # Set fixed param grads to 0.
                self.pruner.do_weight_decay_and_make_grads_zero()

                # Gradient is applied across all ranks
                optimizers.step()
                schedulers.step()

                t.set_postfix({'loss': train_loss.avg.item(),
                               'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                               'lr': '{:.1E}'.format(curr_lrs[0]),
                               'lr_c_a': '{:.1E}'.format(curr_lrs[1]),
                               'sparsity': self.pruner.calculate_sparsity()})
                t.update(1)

        summary = {'loss': '{:.3f}'.format(train_loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                   'lr': '{:.1E}'.format(curr_lrs[0]),
                   'lr_c_a': '{:.1E}'.format(curr_lrs[1]),
                   'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()), }

        if self.args.log_path:
            logging.info(('In train()-> Train Ep. #{} '.format(epoch_idx + 1)
                          + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))

        return train_accuracy.avg.item()

    # {{{ Evaluate classification
    def validate(self, epoch_idx):
        """Performs evaluation."""
        self.pruner.apply_mask()
        self.sample = False
        self.model.eval()
        val_loss = Metric('val_loss')
        val_accuracy = Metric('val_accuracy')

        with tqdm(total=len(self.val_loader),
                  desc='Val Ep. #{}: '.format(epoch_idx + 1),
                  ascii=True) as t:
            with torch.no_grad():
                for info in self.val_loader:
                    if self.args.cuda:
                        input_ids = info['input_ids'].cuda()
                        attention_mask = info['attention_mask'].cuda()
                        target = info['label'].cuda()

                    output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
                    num = input_ids.size(0)
                    val_loss.update(output[0], num)
                    val_accuracy.update(classification_accuracy(output[1], target), num)

                    t.set_postfix({'loss': val_loss.avg.item(),
                                   'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                   'sparsity': self.pruner.calculate_sparsity(),
                                   'task{} ratio'.format(
                                       self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                   'zero ratio': self.pruner.calculate_zero_ratio()})
                    t.update(1)

        summary = {'loss': '{:.3f}'.format(val_loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                   'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()),
                   'task{} ratio'.format(self.inference_dataset_idx): '{:.3f}'.format(
                       self.pruner.calculate_curr_task_ratio()),
                   'zero ratio': '{:.3f}'.format(self.pruner.calculate_zero_ratio())}

        if self.args.log_path:
            logging.info(('In validate()-> Val Ep. #{} '.format(epoch_idx + 1)
                          + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        return val_accuracy.avg.item()

    # }}}

    def one_shot_prune(self, one_shot_prune_perc):
        self.pruner.one_shot_prune(one_shot_prune_perc)
        return

    def save_checkpoint(self, epoch_idx, save_folder):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=epoch_idx + 1)

        for name, module in self.model.module.bert.named_modules():
            if isinstance(module, nl.SharableLinear):
                if module.bias is not None:
                    self.shared_layer_info[self.args.dataset][
                        'bias'][name] = module.bias
            elif isinstance(module, nn.LayerNorm):
                self.shared_layer_info[self.args.dataset][
                    'ln_layer_weight'][name] = module.weight
                self.shared_layer_info[self.args.dataset][
                    'ln_layer_bias'][name] = module.bias

        for name, module in self.model.module.bert.state_dict().items():
            if 'ada' in name:
                self.shared_layer_info[self.args.dataset]['ada'][name] = module

        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),
            'dataset_history': self.model.module.datasets,
            'dataset2num_classes': self.model.module.dataset2num_classes,
            'masks': self.pruner.masks,
            'shared_layer_info': self.shared_layer_info
        }
        torch.save(checkpoint, filepath)
        return

    def load_checkpoint(self, resume_from_epoch, save_folder, args):

        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()
            masks = checkpoint['masks']
            if args.mode == 'finetune':
                for name, param in state_dict.items():
                    if name == 'classifier.weight' or name == 'classifier.bias' or 'LayerNorm' in name or 'saved' in name:
                        continue
                    elif len(curr_model_state_dict[name].size()) == 2 and (True in [ele in name for ele in args.unfreeze_layers]) and not ('ada' in name or 'rho' in name):
                        # SharableLinear
                        curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(torch.where(
                            (masks[name[5:-7]].gt(0) & masks[name[5:-7]].lt(self.inference_dataset_idx)).cuda(),
                            param, curr_model_state_dict[name].cuda()))
                    else:
                        curr_model_state_dict[name].copy_(param)
                self.model.module.bert_saved = deepcopy(self.model.module.bert)
            else:
                for name, param in state_dict.items():
                    curr_model_state_dict[name].copy_(param)
        return

    def load_checkpoint_only_for_evaluate(self, resume_from_epoch, save_folder):

        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()

            for name, param in state_dict.items():
                if 'saved' in name or name == 'classifier.weight' or name == 'classifier.bias' or ('aug' in name and not 'ada' in name):
                    continue
                if 'ada' in name:
                    curr_model_state_dict[name].copy_(self.shared_layer_info[self.args.dataset]['ada'][name[5:]])
                else:
                    curr_model_state_dict[name].copy_(param)

            # load the layer norm params and bias in SharableLinear in correspond to curr dataset
            for name, module in self.model.module.bert.named_modules():
                if isinstance(module, nl.SharableLinear):
                    if module.bias is not None:
                        module.bias = self.shared_layer_info[self.args.dataset]['bias'][name]

                elif isinstance(module, nn.LayerNorm):
                    module.weight = self.shared_layer_info[self.args.dataset][
                        'ln_layer_weight'][name]
                    module.bias = self.shared_layer_info[self.args.dataset][
                        'ln_layer_bias'][name]

        return
