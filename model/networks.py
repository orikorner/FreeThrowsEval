import torch
import torch.nn as nn
import torch.nn.functional as F


# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)
#

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        print("IN LINEAR")
        exit()
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.normal_(m.weight, 0.0, 0.02)
    #     nn.init.constant_(m.bias, 0)


class PrintLayer(nn.Module):
    def __init__(self, layer_name, msg, is_header=False, dbg_mode=False):
        super(PrintLayer, self).__init__()
        self.is_header = is_header
        self.layer_name = layer_name
        self.msg = msg
        self.dbg_mode = dbg_mode

    def forward(self, x):
        if self.dbg_mode is False:
            return x

        if self.is_header:
            print('')
            print('(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?) Header (?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)')
            print(f'(?) {self.layer_name} - {self.msg}\n(?) Data Shape: {x.shape}')
            print('(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)')
        else:
            print(f'(?) {self.layer_name} - {self.msg}: {x.shape}')

        return x


class Encoder(nn.Module):
    def __init__(self, channels, kernel_size=8, global_pool=None, convpool=None, compress=False, dbg_mode=False):
        super(Encoder, self).__init__()

        self.dbg_mode = dbg_mode

        model = []
        acti = nn.LeakyReLU(0.2)
        # acti = nn.Sigmoid()

        nr_layer = len(channels) - 2 if compress else len(channels) - 1

        if dbg_mode:
            dbg_header_msg = f'nr_layer:{nr_layer}, channels:{channels}, compress:{compress}, kernel_size:{kernel_size},\n(?) convpool:{convpool}, global_pool:{global_pool} '
            model.append(PrintLayer(layer_name=f'General Info', msg=dbg_header_msg, is_header=True, dbg_mode=dbg_mode))

        for i in range(nr_layer):
            if convpool is None:
                pad = (kernel_size - 2) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(PrintLayer(layer_name=f'ReflectionPad({i})', msg='Output Shape Is', dbg_mode=dbg_mode))
                model.append(nn.Conv1d(channels[i], channels[i + 1],
                                       kernel_size=kernel_size, stride=2))
                model.append(
                    PrintLayer(layer_name=f'Conv1D({i})', msg=f'Kernel Size={kernel_size}, Stride=2, Output Shape Is',
                               dbg_mode=dbg_mode))
                model.append(acti)
            else:
                pad = (kernel_size - 1) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(PrintLayer(layer_name=f'ReflectionPad({i})', msg='Output Shape Is', dbg_mode=dbg_mode))
                model.append(nn.Conv1d(channels[i], channels[i + 1],
                                       kernel_size=kernel_size, stride=1))
                model.append(
                    PrintLayer(layer_name=f'Conv1D({i})', msg=f'Kernel Size={kernel_size}, Stride=1, Output Shape Is',
                               dbg_mode=dbg_mode))
                model.append(acti)
                model.append(convpool(kernel_size=2, stride=2))
                model.append(PrintLayer(layer_name=f'convpool({i})', msg='Kernel Size=2, Stride=2, Output Shape Is',
                                        dbg_mode=dbg_mode))

        if dbg_mode and global_pool is not None:
            self.glob_pool_print = PrintLayer(layer_name='global_pool', msg='Kernel Size=2, Stride=2, Output Shape Is',
                                              dbg_mode=dbg_mode)

        self.global_pool = global_pool
        self.compress = compress

        self.model = nn.Sequential(*model)

        if self.compress:
            self.conv1x1 = nn.Conv1d(channels[-2], channels[-1], kernel_size=1)
            self.c1x1_print = PrintLayer(layer_name='conv1x1', msg='Kernel Size=1, Stride=default, Output Shape Is',
                                         dbg_mode=dbg_mode)

    def forward(self, x):
        x = self.model(x)
        if self.global_pool is not None:
            ks = x.shape[-1]
            x = self.global_pool(x, ks)
            if self.dbg_mode:
                x = self.glob_pool_print(x)
            if self.compress:
                x = self.conv1x1(x)
                x = self.c1x1_print(x)
        return x


class FtNet(nn.Module):
    """
    This is the ensembled net object - meaning it combines the different part of the network.
    """

    def __init__(self, mot_en_channels, body_en_channels, global_pool=None, convpool=None, compress=False,
                 dbg_mode=False):
        super(FtNet, self).__init__()

        self.dbg_mode = dbg_mode

        self.data_bn = nn.BatchNorm1d(mot_en_channels[0] * 1)
        # model = []
        # here we assert num channles of first part matches second part
        self.mot_encoder = Encoder(mot_en_channels, kernel_size=5, dbg_mode=dbg_mode)
        self.static_encoder = Encoder(body_en_channels, kernel_size=5, global_pool=global_pool, convpool=convpool,
                                      compress=compress, dbg_mode=dbg_mode)
        # for name, param in self.mot_encoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        # self.mot_encoder.apply(init_weights)
        # for name, param in self.mot_encoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        # exit()
        # self.static_encoder.apply(init_weights)
        self.mot_pooling = nn.AdaptiveAvgPool2d((None, 4))
        # self.conv2_out = nn.Conv2d(768, 2, kernel_size=1)
        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 48)
        self.fc3 = nn.Linear(48, 2)

        self.mot_encoder_print = PrintLayer(layer_name='Motion Encoder', msg='Output Shape Is', dbg_mode=dbg_mode)
        self.mot_pooling_print = PrintLayer(layer_name='Motion Encoder Pooling', msg='Output Shape Is', dbg_mode=dbg_mode)
        self.static_encoder_print = PrintLayer(layer_name='Static Encoder', msg='Output Shape Is', dbg_mode=dbg_mode)
        self.fc1_print = PrintLayer(layer_name='Fully Connected 1', msg='Output Shape Is', dbg_mode=dbg_mode)
        self.fc2_print = PrintLayer(layer_name='Fully Connected 2', msg='Output Shape Is', dbg_mode=dbg_mode)
        self.fc3_print = PrintLayer(layer_name='Fully Connected 3', msg='Output Shape Is', dbg_mode=dbg_mode)
        self.lin_act = F.relu
        # self.lin_act = nn.Sigmoid()
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
        x = self.data_bn(x)

        mot_out = self.mot_encoder(x)
        mot_out = self.mot_encoder_print(mot_out)
        # print(f'm shape is: {mot_out.shape}')
        # mot_out = self.mot_pooling(mot_out)
        # mot_out = self.mot_pooling_print(mot_out)
        # print(f'm shape is: {mot_out.shape}')
        stat_out = self.static_encoder(x[:, :-2, :])  # 2 coords belong to Velocity of hips between consecutive frames
        stat_out = self.static_encoder_print(stat_out)
        # print(f'stat_out shape is: {stat_out.shape}')
        stat_out = stat_out.repeat(1, 1, mot_out.shape[-1])
        # print(f'stat_out after repeat shape is: {stat_out.shape}')
        feat_map = torch.cat([mot_out, stat_out], dim=1)
        # print(f'd shape is: {feat_map.shape}')

        feat_map = feat_map.view(-1, 768)
        # print(f'd shape after view is: {feat_map.shape}')
        fc_out = self.lin_act(self.fc1(feat_map))
        fc_out = self.fc1_print(fc_out)
        fc_out = self.lin_act(self.fc2(fc_out))
        fc_out = self.fc2_print(fc_out)
        fc_out = self.fc3(fc_out)
        fc_out = self.fc3_print(fc_out)
        # feat_map = feat_map.reshape(-1, 768)
        # feat_map.unsqueeze_(2)
        # feat_map.unsqueeze_(3)
        # fc_out = self.conv2_out(feat_map)
        # fc_out.squeeze_(3)
        # fc_out.squeeze_(2)


        return fc_out
