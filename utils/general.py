import os
import os.path as osp


class TrainClock(object):
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']


def ensure_dir(fpath):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not osp.exists(fpath):
        os.makedirs(fpath)


def ensure_dirs(fpath_list):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(fpath_list, list) and not isinstance(fpath_list, str):
        for fpath in fpath_list:
            ensure_dir(fpath)
    else:
        ensure_dir(fpath_list)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
