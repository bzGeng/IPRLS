"""Handles all the pruning-related stuff."""
import torch
import models.layers as nl
import sys


class SparsePruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, masks, args, inference_dataset_idx):
        self.model = model
        self.args = args
        self.sparsity_func_exponent = 3
        self.masks = masks

        if args.mode == 'prune' or args.mode == 'inference':
            self.current_dataset_idx = self.model.module.datasets.index(args.dataset) + 1
        elif args.mode == 'finetune':
            self.current_dataset_idx = len(self.model.module.datasets) - 1
        else:
            print('We do not support \'{}\' mode'.format(args.mode))
            sys.exit(-1)

        self.inference_dataset_idx = inference_dataset_idx
        self.model.module.current_dataset_idx = self.inference_dataset_idx
        return

    def one_shot_prune(self, one_shot_prune_perc):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
        print('Pruning for dataset idx: %d' % (self.current_dataset_idx))
        print('Pruning each layer by removing %.2f%% of values' % (100 * one_shot_prune_perc))

        for (name, module), (name_saved, module_saved) in zip(self.model.module.bert.named_modules(),
                                                              self.model.module.bert_saved.named_modules()):
            if isinstance(module, nl.SharableLinear) and (True in [ele in name for ele in self.args.shared_layers]):
                assert name == name_saved
                mask = self.masks[name]
                trainer_weight = module.weight
                out_features, in_features = trainer_weight.shape
                trainer_weight_sigma = torch.log1p(torch.exp(module.weight_rho))
                trainer_weight_sigma = trainer_weight_sigma.expand(out_features, in_features)
                prune_score = trainer_weight / trainer_weight_sigma

                tensor = prune_score[mask.eq(self.current_dataset_idx) | mask.eq(0)]  # This will flatten weights
                abs_tensor = tensor.abs()
                cutoff_rank = round(one_shot_prune_perc * tensor.numel())
                cutoff_value = abs_tensor.cpu().kthvalue(cutoff_rank)[0].cuda()  # value at cutoff rank

                remove_mask = prune_score.abs().le(cutoff_value) * mask.eq(self.current_dataset_idx).cuda()
                # mask = 1 - remove_mask
                mask[remove_mask.eq(1)] = 0
        return

    def calculate_sparsity(self):
        total_elem = 0
        zero_elem = 0

        for name, module in self.model.module.bert.named_modules():
            if isinstance(module, nl.SharableLinear) and (True in [ele in name for ele in self.args.shared_layers]):
                mask = self.masks[name]
                total_elem += torch.sum(mask.eq(self.inference_dataset_idx) | mask.eq(0))
                zero_elem += torch.sum(mask.eq(0))
        if total_elem.cpu() != 0.0:
            return float(zero_elem.cpu()) / float(total_elem.cpu())
        else:
            return 0.0

    def calculate_curr_task_ratio(self):
        total_elem = 0
        curr_task_elem = 0

        for name, module in self.model.module.bert.named_modules():
            if isinstance(module, nl.SharableLinear) and (True in [ele in name for ele in self.args.shared_layers]):
                mask = self.masks[name]
                total_elem += mask.numel()
                curr_task_elem += torch.sum(mask.eq(self.inference_dataset_idx))

        return float(curr_task_elem.cpu()) / total_elem

    def calculate_zero_ratio(self):
        total_elem = 0
        zero_elem = 0

        for name, module in self.model.module.bert.named_modules():
            if isinstance(module, nl.SharableLinear) and (True in [ele in name for ele in self.args.shared_layers]):
                mask = self.masks[name]
                total_elem += mask.numel()
                zero_elem += torch.sum(mask.eq(0))

        return float(zero_elem.cpu()) / total_elem

    def do_weight_decay_and_make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.masks
        for name, module in self.model.module.bert.named_modules():
            if isinstance(module, nl.SharableLinear) and (True in [ele in name for ele in self.args.shared_layers]):
                mask = self.masks[name]
                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data.add_(self.args.weight_decay, module.weight.data)
                    if self.args.mode == 'prune':
                        module.weight.grad.data[mask.ne(self.current_dataset_idx)] = 0
        return

    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.masks

        for name, module in self.model.module.bert.named_modules():
            if isinstance(module, nl.SharableLinear) and (True in [ele in name for ele in self.args.shared_layers]):
                layer_mask = self.masks[name]
                module.weight.data[layer_mask.eq(0)] = 0.0
        return

    def apply_mask(self):
        """To be done to retrieve weights just for a particular dataset."""
        for name, module in self.model.module.bert.named_modules():
            if isinstance(module, nl.SharableLinear) and (True in [ele in name for ele in self.args.shared_layers]):
                weight = module.weight.data
                mask = self.masks[name].cuda()
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(self.inference_dataset_idx)] = 0.0
        return

    def make_finetuning_mask(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.masks
        self.current_dataset_idx += 1

        for name, module in self.model.module.bert.named_modules():
            if isinstance(module, nl.SharableLinear) and (True in [ele in name for ele in self.args.shared_layers]):
                mask = self.masks[name]
                mask[mask.eq(0)] = self.current_dataset_idx