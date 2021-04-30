import torch
from torch.utils.data import Dataset


def Amason_train_loader(dataset_path, train_batch_size, args, pin_memory=True):
    train_set = torch.load(dataset_path)
    return torch.utils.data.DataLoader(train_set,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=args.workers, pin_memory=pin_memory)


def Amason_val_loader(dataset_path, val_batch_size, args, pin_memory=True):
    val_set = torch.load(dataset_path)
    return torch.utils.data.DataLoader(val_set,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=args.workers, pin_memory=pin_memory)
