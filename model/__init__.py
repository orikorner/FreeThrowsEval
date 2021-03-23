from model.networks import FtNet
import torch.nn as nn
import torch.nn.functional as F


def get_network(config):
    assert config.name is not None
    if config.name == 'skeleton':
        return FtNet(mot_en_channels=config.mot_en_channels, body_en_channels=config.body_en_channels,
                     global_pool=F.max_pool1d, convpool=nn.MaxPool1d, compress=False)
