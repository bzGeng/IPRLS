import json
import math
import os
import random
import pdb
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import logging

import utils
from utils import Optimizers, Schedulers, set_logger
from utils.build_data import build_data_seperate, build_data_file
import utils.Amazon_dataset as dataset
import models
import models.layers as nl
from models.Bert_model import BertConfig

from transformers import get_cosine_schedule_with_warmup
from transformers.optimization import AdamW


def settings(args):
    if args.save_folder and not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.log_path:
        set_logger(args.log_path)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        args.cuda = False

    cudnn.benchmark = True

    # Set default train and test path if not provided as input.
    utils.set_dataset_paths(args)

    args.unfreeze_layers = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.',
                            'layer.7.', 'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.', 'pooler']
    args.shared_layers = args.unfreeze_layers
    # preprocess
    if args.build_data_seperate:
        build_data_seperate()

    if args.mode == 'finetune' and args.build_data_file:
        build_data_file(args)


def set_seeds(args):
    # set all possible seeds
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def check_resume_epoch(args):
    resume_from_epoch = 0
    resume_folder = args.load_folder
    for try_epoch in range(200, 0, -1):
        if os.path.exists(args.checkpoint_format.format(
                save_folder=resume_folder, epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break
    if args.restore_epoch:
        resume_from_epoch = args.restore_epoch
    logging.info('resume_from_epoch {}'.format(resume_from_epoch))
    return resume_from_epoch, resume_folder


def info_reload(resume_from_epoch, args):
    if resume_from_epoch:
        filepath = args.checkpoint_format.format(save_folder=args.load_folder, epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        checkpoint_keys = checkpoint.keys()
        dataset_history = checkpoint['dataset_history']
        dataset2num_classes = checkpoint['dataset2num_classes']
        masks = checkpoint['masks']
        if 'shared_layer_info' in checkpoint_keys:
            shared_layer_info = checkpoint['shared_layer_info']
        else:
            shared_layer_info = {}
    else:
        dataset_history = []
        dataset2num_classes = {}
        masks = {}
        shared_layer_info = {}
    return dataset_history, dataset2num_classes, masks, shared_layer_info


def build_model(args, dataset_history, dataset2num_classes):
    if args.approach == 'IPRLS':
        args.Bert_config_path = args.Bert_path + 'config_prfs.json'
        args.configuration = BertConfig.from_json_file(args.Bert_config_path)
        model = models.__dict__[args.approach](dataset_history=dataset_history,
                                               dataset2num_classes=dataset2num_classes,
                                               args=args)
    else:
        print('Error!')
        sys.exit(0)

    return model


def load_or_build_masks(masks, model, args):
    if not masks:
        for name, module in model.bert.named_modules():
            if 'ada' in name:
                continue
            if isinstance(module, nl.SharableLinear) and (True in [ele in name for ele in args.shared_layers]):
                mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                if 'cuda' in module.weight.data.type():
                    mask = mask.cuda()
                masks[name] = mask
    return masks


def load_data(args):
    train_loader = dataset.Amason_train_loader(args.train_path, args.batch_size, args)
    val_loader = dataset.Amason_val_loader(args.val_path, args.val_batch_size, args)
    test_loader = dataset.Amason_val_loader(args.test_path, args.val_batch_size, args)
    return train_loader, val_loader, test_loader


def calculate_start_epoch(args, resume_from_epoch):
    if args.save_folder != args.load_folder:
        start_epoch = 0
    else:
        start_epoch = resume_from_epoch
    return start_epoch


def set_optimizers(args, model):
    lr = args.lr
    if args.mode == 'finetune':
        lr_adapters = args.lr_adapters
    else:
        lr_adapters = args.lr_adapters * 0.1
    # update all layers
    named_params = dict(model.named_parameters())
    params_to_optimize_via_AdamW = []
    named_params_to_optimize_via_AdamW = []
    params_to_optimize_via_AdamW2 = []
    named_params_to_optimize_via_AdamW2 = []

    for name, param in named_params.items():
        if 'bert_saved' in name:
            continue
        if 'classifiers' in name:
            if '.{}.'.format(model.module.datasets.index(args.dataset)) in name:
                params_to_optimize_via_AdamW2.append(param)
                named_params_to_optimize_via_AdamW2.append(name)
            continue
        elif 'ada' in name:
            params_to_optimize_via_AdamW.append(param)
            named_params_to_optimize_via_AdamW.append(name)
            continue
        elif not (True in [ele in name for ele in args.shared_layers]):
            continue
        else:
            params_to_optimize_via_AdamW2.append(param)
            named_params_to_optimize_via_AdamW2.append(name)

    lr_adapters = AdamW(params_to_optimize_via_AdamW, lr=lr_adapters,
                        weight_decay=1e-8)
    optimizer_network = AdamW(params_to_optimize_via_AdamW2, lr=lr,
                              weight_decay=0.0)
    optimizers = Optimizers()
    optimizers.add(optimizer_network, lr)
    optimizers.add(lr_adapters, lr_adapters)

    scheduler_network = None
    if args.mode == 'finetune':
        scheduler_network = get_cosine_schedule_with_warmup(optimizer_network, 0, args.training_steps)
    elif args.mode == 'prune':
        scheduler_network = get_cosine_schedule_with_warmup(optimizer_network, 0, args.training_steps)
    scheduler_adapters = get_cosine_schedule_with_warmup(lr_adapters, 0, int(args.training_steps * 1.1))
    schedulers = Schedulers()
    schedulers.add(scheduler_network)
    schedulers.add(scheduler_adapters)

    return optimizers, schedulers


def check_if_need_remove_checkpoint_files(args):
    if args.save_folder is not None:
        paths = os.listdir(args.save_folder)
        if paths and '.pth.tar' in paths[0]:
            for checkpoint_file in paths:
                os.remove(os.path.join(args.save_folder, checkpoint_file))
    else:
        print('Something is wrong! Block the program with pdb')
        pdb.set_trace()


def check_if_need_build_shared_layer_info(args, shared_layer_info):
    if args.dataset not in shared_layer_info:
        shared_layer_info[args.dataset] = {
            'bias': {},
            'ln_layer_weight': {},
            'ln_layer_bias': {},
            'ada': {}
        }
    return shared_layer_info


def obtain_curr_lrs(optimizers):
    curr_lrs = []
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            curr_lrs.append(param_group['lr'])
            break
    return curr_lrs


def freeze_modules(model, args):
    classifier_name = 'classifiers' + '.' + str(model.module.datasets.index(args.dataset))
    unfreeze_layers = args.unfreeze_layers + [classifier_name]

    '''
    for name, param in model.named_parameters():
        print(name, param.size())
    print("*" * 30)
    print('\n')
    '''
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'bert_saved' in name:
            continue
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    '''   # verification
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
    '''


def write_log(args, avg_val_acc):
    if args.logfile:
        json_data = {}
        if os.path.isfile(args.logfile):
            with open(args.logfile) as json_file:
                json_data = json.load(json_file)
        json_data[args.dataset] = '{:.4f}'.format(avg_val_acc)
        with open(args.logfile, 'w') as json_file:
            json.dump(json_data, json_file)


def write_acc(args, avg_val_acc):
    if args.acc_log_path:
        with open(args.acc_log_path) as json_file:
            json_data = json.load(json_file)
        json_data[args.dataset] = '{:.4f}'.format(avg_val_acc)
        with open(args.acc_log_path, 'w') as json_file:
            json.dump(json_data, json_file)
