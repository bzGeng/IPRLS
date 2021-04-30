import torch.nn as nn

from utils import argparser
from utils.manager import Manager
from utils.settings import *


def main():
    args = argparser.get_parser()
    set_seeds(args)
    settings(args)

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch, resume_folder = check_resume_epoch(args)

    dataset_history, dataset2num_classes, masks, shared_layer_info = info_reload(resume_from_epoch, args)

    model = build_model(args, dataset_history, dataset2num_classes)

    # Add and set the model dataset.
    model.add_dataset(args.dataset, args.num_classes)
    model.set_dataset(args.dataset)

    masks = load_or_build_masks(masks, model, args)

    model = nn.DataParallel(model)

    shared_layer_info = check_if_need_build_shared_layer_info(args, shared_layer_info)

    train_loader, val_loader, test_loader = load_data(args)

    # if we are going to save checkpoint in other folder, then we recalculate the starting epoch
    start_epoch = calculate_start_epoch(args, resume_from_epoch)

    manager = Manager(args, model, shared_layer_info, masks, train_loader, val_loader)

    args.training_steps = args.epochs * len(train_loader)
    optimizers, schedulers = set_optimizers(args, model)
    # manager.save_checkpoint(optimizers, 0, args.save_folder)
    manager.load_checkpoint(resume_from_epoch, resume_folder, args)

    """Performs training."""
    curr_lrs = obtain_curr_lrs(optimizers)

    if args.mode == 'prune':
        print('Sparsity ratio: {}'.format(args.one_shot_prune_perc))
        print('Execute one shot pruning ...')
        manager.one_shot_prune(args.one_shot_prune_perc)
        manager.pruner.apply_mask()
    elif args.mode == 'finetune':
        manager.pruner.make_finetuning_mask()
        logging.info('Finetune stage...')

    freeze_modules(model, args)

    max_val_acc = 0
    max_test_acc = 0
    model = model.cuda()
    for epoch_idx in range(start_epoch, args.epochs):
        need_save = False
        manager.train(optimizers, schedulers, epoch_idx, curr_lrs)
        avg_val_acc = manager.validate(epoch_idx)
        manager.val_loader = test_loader
        logging.info("performance on test")
        test_acc = manager.validate(epoch_idx)
        manager.val_loader = val_loader
        if avg_val_acc >= max_val_acc:
            need_save = True
            max_val_acc = avg_val_acc
            max_test_acc = test_acc

        if need_save:
            check_if_need_remove_checkpoint_files(args)
            manager.save_checkpoint(epoch_idx, args.save_folder)
    logging.info('-' * 16)

    # if args.mode == 'prune':
    #     write_acc(args, max_test_acc)


if __name__ == '__main__':
    main()
