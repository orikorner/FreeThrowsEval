import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, channels, kernel_size=8, global_pool=None, convpool=None, compress=False):
        super(Encoder, self).__init__()

        model = []
        acti = nn.LeakyReLU(0.2)

        nr_layer = len(channels) - 2 if compress else len(channels) - 1

        for i in range(nr_layer):
            if convpool is None:
                pad = (kernel_size - 2) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i + 1],
                                       kernel_size=kernel_size, stride=2))
                model.append(acti)
            else:
                pad = (kernel_size - 1) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i + 1],
                                       kernel_size=kernel_size, stride=1))
                model.append(acti)
                model.append(convpool(kernel_size=2, stride=2))

        self.global_pool = global_pool
        self.compress = compress

        self.model = nn.Sequential(*model)

        if self.compress:
            self.conv1x1 = nn.Conv1d(channels[-2], channels[-1], kernel_size=1)

    def forward(self, x):
        print('=========== forward shape in encoder.forward =======')
        print(x.shape)
        x = self.model(x)
        if self.global_pool is not None:
            ks = x.shape[-1]
            x = self.global_pool(x, ks)
            if self.compress:
                x = self.conv1x1(x)
        return x


class FtNet(nn.Module):
    """
    This is the ensembled net object - meaning it combines the different part of the network.
    """

    def __init__(self, body_en_channels, global_pool=None, convpool=None, compress=False):
        super(FtNet, self).__init__()

        # here we assert num channles of first part matches second part
        self.static_encoder = Encoder(body_en_channels, kernel_size=7, global_pool=global_pool, convpool=convpool,
                                      compress=compress)

    # def cross(self, x1, x2):
    #     m1 = self.mot_encoder(x1)
    #     b1 = self.static_encoder(x1[:, :-2, :])
    #     m2 = self.mot_encoder(x2)
    #     b2 = self.static_encoder(x2[:, :-2, :])
    #
    #     out1 = self.decoder(torch.cat([m1, b1.repeat(1, 1, m1.shape[-1])], dim=1))
    #     out2 = self.decoder(torch.cat([m2, b2.repeat(1, 1, m2.shape[-1])], dim=1))
    #     out12 = self.decoder(torch.cat([m1, b2.repeat(1, 1, m1.shape[-1])], dim=1))
    #     out21 = self.decoder(torch.cat([m2, b1.repeat(1, 1, m2.shape[-1])], dim=1))
    #
    #     return out1, out2, out12, out21

    # def transfer(self, x1, x2):
    #     m1 = self.mot_encoder(x1)
    #     b2 = self.static_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])
    #
    #     out12 = self.decoder(torch.cat([m1, b2], dim=1))
    #
    #     return out12
    #
    # def cross_with_triplet(self, x1, x2, x12, x21):
    #     m1 = self.mot_encoder(x1)
    #     b1 = self.static_encoder(x1[:, :-2, :])
    #     m2 = self.mot_encoder(x2)
    #     b2 = self.static_encoder(x2[:, :-2, :])
    #
    #     out1 = self.decoder(torch.cat([m1, b1.repeat(1, 1, m1.shape[-1])], dim=1))
    #     out2 = self.decoder(torch.cat([m2, b2.repeat(1, 1, m2.shape[-1])], dim=1))
    #     out12 = self.decoder(torch.cat([m1, b2.repeat(1, 1, m1.shape[-1])], dim=1))
    #     out21 = self.decoder(torch.cat([m2, b1.repeat(1, 1, m2.shape[-1])], dim=1))
    #
    #     m12 = self.mot_encoder(x12)
    #     b12 = self.static_encoder(x12[:, :-2, :])
    #     m21 = self.mot_encoder(x21)
    #     b21 = self.static_encoder(x21[:, :-2, :])
    #
    #     outputs = [out1, out2, out12, out21]
    #     motionvecs = [m1.reshape(m1.shape[0], -1),
    #                   m2.reshape(m2.shape[0], -1),
    #                   m12.reshape(m12.shape[0], -1),
    #                   m21.reshape(m21.shape[0], -1)]
    #     bodyvecs = [b1.reshape(b1.shape[0], -1),
    #                   b2.reshape(b2.shape[0], -1),
    #                   b21.reshape(b21.shape[0], -1),
    #                   b12.reshape(b12.shape[0], -1)]
    #
    #     return outputs, motionvecs, bodyvecs

    def forward(self, x):
        # m = self.mot_encoder(x)
        print('============ forward of network ========== ')
        print(x.shape)
        print('============ = ========== ')
        b = self.static_encoder(x[:, :-2, :])
        # b = b.repeat(1, 1, m.shape[-1])
        # d = torch.cat([m, b], dim=1)
        # d = self.decoder(d)
        # return d
        return b
