"""Main entry point for doing all stuff."""
import warnings

import torch.nn as nn
from utils import argparser
from utils.manager import Manager
from utils.settings import *
from utils.logger import CsvLogger

# To prevent PIL warnings.
warnings.filterwarnings("ignore")


def main():
    """Do stuff."""
    args = argparser.get_parser()

    # Don't use this, neither set learning rate as a linear function
    # of the count of gpus, it will make accuracy lower
    # args.batch_size = args.batch_size * torch.cuda.device_count()
    set_seeds(args)
    settings(args)
    logger = CsvLogger(file_name='acc', resume=True, path='results', data_format='csv')
    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch, resume_folder = check_resume_epoch(args)

    dataset_history, dataset2num_classes, masks, shared_layer_info = info_reload(resume_from_epoch, args)

    model = build_model(args, dataset_history, dataset2num_classes)
    model = nn.DataParallel(model)
    for dataset in dataset_history:
        args.dataset = dataset
        utils.set_dataset_paths(args)
        model.module.set_dataset(args.dataset)
        train_loader, val_loader, test_loader = load_data(args)
        manager = Manager(args, model, shared_layer_info, masks, train_loader, test_loader)
        manager.load_checkpoint_only_for_evaluate(resume_from_epoch, resume_folder)
        model = model.cuda()
        test_acc = manager.validate(resume_from_epoch - 1)
        idx = dataset_history.index(dataset)
        finished = len(dataset_history)
        logger.add(dataset=dataset, idx=idx, finished=finished, acc=round(test_acc, 4))
        logger.save()
    return


if __name__ == '__main__':
    main()
