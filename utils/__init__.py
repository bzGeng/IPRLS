import sys
import torch
import logging


class Optimizers(object):
    def __init__(self):
        self.optimizers = []
        self.lrs = []

    def add(self, optimizer, lr):
        self.optimizers.append(optimizer)
        self.lrs.append(lr)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def __getitem__(self, index):
        return self.optimizers[index]

    def __setitem__(self, index, value):
        self.optimizers[index] = value


class Schedulers(object):
    def __init__(self):
        self.schedulers = []

    def add(self, scheduler):
        self.schedulers.append(scheduler)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def __getitem__(self, index):
        return self.schedulers[index]

    def __setitem__(self, index, value):
        self.schedulers[index] = value


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        return

    def update(self, val, num):
        self.sum += val.mean() * num
        self.n += num

    @property
    def avg(self):
        return self.sum / self.n


def classification_accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def set_dataset_paths(args):
    args.train_path = 'data/file/%s.train.file' % args.dataset
    args.val_path = 'data/file/%s.val.file' % args.dataset
    args.test_path = 'data/file/%s.test.file' % args.dataset


def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return
