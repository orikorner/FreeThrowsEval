from model.networks import FtNet
import torch.nn as nn
import torch.nn.functional as F


def get_network(config):
    # assert config.name is not None
    # if config.name == 'skeleton':
    if True:
        return FtNet(body_en_channels=config.body_en_channels,
                     global_pool=F.max_pool1d, convpool=nn.MaxPool1d, compress=False)
