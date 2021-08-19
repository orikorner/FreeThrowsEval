# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
#
# class PrintLayer(nn.Module):
#     def __init__(self, layer_name, msg, is_header=False, dbg_mode=False):
#         super(PrintLayer, self).__init__()
#         self.is_header = is_header
#         self.layer_name = layer_name
#         self.msg = msg
#         self.dbg_mode = dbg_mode
#
#     def forward(self, x):
#         if self.dbg_mode is False:
#             return x
#
#         if self.is_header:
#             print('')
#             print('(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?) Header (?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)')
#             print(f'(?) {self.layer_name} - {self.msg}\n(?) Data Shape: {x.shape}')
#             print('(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)(?)')
#         else:
#             print(f'(?) {self.layer_name} - {self.msg}: {x.shape}')
#
#         return x
#
#
# class Encoder(nn.Module):
#     def __init__(self, channels, kernel_size=8, global_pool=None, convpool=None, compress=False, dbg_mode=False):
#         super(Encoder, self).__init__()
#
#         self.dbg_mode = dbg_mode
#
#         model = []
#         acti = nn.LeakyReLU(0.2)
#         # acti = nn.Sigmoid()
#
#         nr_layer = len(channels) - 2 if compress else len(channels) - 1
#
#         if dbg_mode:
#             dbg_header_msg = f'nr_layer:{nr_layer}, channels:{channels}, compress:{compress}, kernel_size:{kernel_size},\n(?) convpool:{convpool}, global_pool:{global_pool} '
#             model.append(PrintLayer(layer_name=f'General Info', msg=dbg_header_msg, is_header=True))
#
#         for i in range(nr_layer):
#             if convpool is None:
#                 pad = (kernel_size - 2) // 2
#                 model.append(nn.ReflectionPad1d(pad))
#                 model.append(PrintLayer(layer_name=f'ReflectionPad({i})', msg='Output Shape Is', dbg_mode=dbg_mode))
#                 model.append(nn.Conv1d(channels[i], channels[i + 1],
#                                        kernel_size=kernel_size, stride=2))
#                 model.append(
#                     PrintLayer(layer_name=f'Conv1D({i})', msg=f'Kernel Size={kernel_size}, Stride=2, Output Shape Is',
#                                dbg_mode=dbg_mode))
#                 model.append(acti)
#             else:
#                 pad = (kernel_size - 1) // 2
#                 model.append(nn.ReflectionPad1d(pad))
#                 model.append(PrintLayer(layer_name=f'ReflectionPad({i})', msg='Output Shape Is', dbg_mode=dbg_mode))
#                 model.append(nn.Conv1d(channels[i], channels[i + 1],
#                                        kernel_size=kernel_size, stride=1))
#                 model.append(
#                     PrintLayer(layer_name=f'Conv1D({i})', msg=f'Kernel Size={kernel_size}, Stride=1, Output Shape Is',
#                                dbg_mode=dbg_mode))
#                 model.append(acti)
#                 model.append(convpool(kernel_size=2, stride=2))
#                 model.append(PrintLayer(layer_name=f'convpool({i})', msg='Kernel Size=2, Stride=2, Output Shape Is',
#                                         dbg_mode=dbg_mode))
#
#         if dbg_mode and global_pool is not None:
#             self.glob_pool_print = PrintLayer(layer_name='global_pool', msg='Kernel Size=2, Stride=2, Output Shape Is',
#                                               dbg_mode=dbg_mode)
#
#         self.global_pool = global_pool
#         self.compress = compress
#
#         self.model = nn.Sequential(*model)
#
#         if self.compress:
#             self.conv1x1 = nn.Conv1d(channels[-2], channels[-1], kernel_size=1)
#             self.c1x1_print = PrintLayer(layer_name='conv1x1', msg='Kernel Size=1, Stride=default, Output Shape Is',
#                                          dbg_mode=dbg_mode)
#
#     def forward(self, x):
#         x = self.model(x)
#         if self.global_pool is not None:
#             ks = x.shape[-1]
#             x = self.global_pool(x, ks)
#             if self.dbg_mode:
#                 x = self.glob_pool_print(x)
#             if self.compress:
#                 x = self.conv1x1(x)
#                 x = self.c1x1_print(x)
#         return x
#
#
# class FtNet(nn.Module):
#     """
#     This is the ensembled net object - meaning it combines the different part of the network.
#     """
#
#     def __init__(self, mot_en_channels, body_en_channels, cls_head_dims,
#                  global_pool=None, convpool=None, compress=False, dbg_mode=False):
#         super(FtNet, self).__init__()
#
#         self.dbg_mode = dbg_mode
#
#         self.data_bn = nn.BatchNorm1d(mot_en_channels[0])
#         self.mot_encoder = Encoder(mot_en_channels, kernel_size=5, dbg_mode=dbg_mode)
#         self.static_encoder = Encoder(body_en_channels, kernel_size=5, global_pool=global_pool, convpool=convpool,
#                                             compress=compress, dbg_mode=dbg_mode)
#
#         self.fc1 = nn.Linear(768, 192)
#         self.fc2 = nn.Linear(192, 48)
#         self.fc3 = nn.Linear(48, 2)
#
#         self.mot_encoder_print = PrintLayer(layer_name='Motion Encoder', msg='Output Shape Is', dbg_mode=dbg_mode)
#         self.static_encoder_print = PrintLayer(layer_name='Static Encoder', msg='Output Shape Is', dbg_mode=dbg_mode)
#         # self.fc1_print = PrintLayer(layer_name='Fully Connected 1', msg='Output Shape Is', dbg_mode=dbg_mode)
#         # self.fc2_print = PrintLayer(layer_name='Fully Connected 2', msg='Output Shape Is', dbg_mode=dbg_mode)
#         # self.fc3_print = PrintLayer(layer_name='Fully Connected 3', msg='Output Shape Is', dbg_mode=dbg_mode)
#         self.lin_act = F.relu
#
#     def forward(self, x):
#         x = self.data_bn(x)
#         mot_out = self.mot_encoder(x)
#         mot_out = self.mot_encoder_print(mot_out)
#
#         stat_out = self.static_encoder(x[:, :-2, :])
#         stat_out = self.static_encoder_print(stat_out)
#
#         stat_out = stat_out.repeat(1, 1, mot_out.shape[-1])
#         feat_map = torch.cat([mot_out, stat_out], dim=1)
#         feat_map = feat_map.view(-1, 768)
#
#         fc_out = self.lin_act(self.fc1(feat_map))
#         # fc_out = self.fc1_print(fc_out)
#         fc_out = self.lin_act(self.fc2(fc_out))
#         # fc_out = self.fc2_print(fc_out)
#         fc_out = self.fc3(fc_out)
#         # fc_out = self.fc3_print(fc_out)
#         return fc_out
#
#     def load_my_state_dict(self, state_dict):
#
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name not in own_state or own_state[name].shape != param.data.shape:
#                 print(name)
#                 continue
#             if isinstance(param, nn.parameter.Parameter):
#                 # backwards compatibility for serialized parameters
#                 param = param.data
#
#             own_state[name].copy_(param)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        m.bias.data.fill_(0.01)
    # elif isinstance(m, nn.Linear):
    #     nn.init.xavier_uniform_(m.weight)
    #     m.bias.data.fill_(0.01)
    # elif isinstance(m, nn.BatchNorm1d):
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


class TrajectoryHead(nn.Module):
    def __init__(self, dims, dbg_mode=False):
        super(TrajectoryHead, self).__init__()

        self.dbg_mode = dbg_mode

        model = []
        nr_layer = len(dims) - 2
        for i in range(nr_layer):
            model.append(nn.Linear(dims[i], dims[i + 1]))
            model.append(nn.ReLU(True))
            # model.append(nn.Dropout())
            model.append(PrintLayer(layer_name=f'Fully Connected {i + 1}', msg='Output Shape Is', dbg_mode=dbg_mode))

        model.append(nn.Linear(dims[-2], dims[-1]))
        # model.append(nn.Dropout())
        model.append(PrintLayer(layer_name=f'Fully Connected {len(dims) - 1}', msg='Output Shape Is', dbg_mode=dbg_mode))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Head(nn.Module):
    def __init__(self, dims, dbg_mode=False):
        super(Head, self).__init__()

        self.dbg_mode = dbg_mode

        model = []
        nr_layer = len(dims) - 2
        for i in range(nr_layer):
            model.append(nn.Linear(dims[i], dims[i + 1]))
            model.append(nn.ReLU(True))
            # model.append(nn.Dropout(p=0.2))
            model.append(PrintLayer(layer_name=f'Fully Connected {i + 1}', msg='Output Shape Is', dbg_mode=dbg_mode))

        model.append(nn.Linear(dims[-2], dims[-1]))
        # model.append(nn.Dropout())
        model.append(PrintLayer(layer_name=f'Fully Connected {len(dims) - 1}', msg='Output Shape Is', dbg_mode=dbg_mode))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class StaticEncoder(nn.Module):
    def __init__(self, channels, kernel_size=5,
                 global_pool=F.max_pool1d, convpool=nn.MaxPool1d, compress=False, dbg_mode=False):
        super(StaticEncoder, self).__init__()

        self.dbg_mode = dbg_mode

        model = []
        acti = nn.LeakyReLU(0.2)
        nr_layer = len(channels) - 2 if compress else len(channels) - 1

        if dbg_mode:
            dbg_header_msg = f'nr_layer:{nr_layer}, channels:{channels}, compress:{compress}, kernel_size:{kernel_size},\n(?) convpool:{convpool}, global_pool:{global_pool} '
            model.append(PrintLayer(layer_name=f'General Info', msg=dbg_header_msg, is_header=True, dbg_mode=dbg_mode))

        # model.append(nn.Sigmoid())
        model.append(nn.BatchNorm1d(channels[0]))
        for i in range(nr_layer):
            pad = (kernel_size - 1) // 2
            model.append(nn.ReflectionPad1d(pad))
            model.append(PrintLayer(layer_name=f'ReflectionPad({i})', msg='Output Shape Is', dbg_mode=dbg_mode))
            model.append(nn.Conv1d(channels[i], channels[i + 1],
                                   kernel_size=kernel_size, stride=1))
            model.append(PrintLayer(layer_name=f'Conv1D({i})',
                                    msg=f'Kernel Size={kernel_size}, Stride=1, Output Shape Is', dbg_mode=dbg_mode))
            model.append(acti)
            model.append(convpool(kernel_size=2, stride=2))
            model.append(PrintLayer(layer_name=f'convpool({i})',
                                    msg='Kernel Size=2, Stride=2, Output Shape Is', dbg_mode=dbg_mode))

        if dbg_mode and global_pool is not None:
            self.glob_pool_print = PrintLayer(layer_name='global_pool',
                                              msg='Kernel Size=2, Stride=2, Output Shape Is', dbg_mode=dbg_mode)

        self.global_pool = global_pool
        # self.global_pool = nn.MaxPool1d(kernel_size=5)
        self.compress = compress

        self.model = nn.Sequential(*model)

        if self.compress:
            self.conv1x1 = nn.Conv1d(channels[-2], channels[-1], kernel_size=1)
            self.c1x1_print = PrintLayer(layer_name='conv1x1',
                                         msg='Kernel Size=1, Stride=default, Output Shape Is', dbg_mode=dbg_mode)

    def forward(self, x):
        x = self.model(x)
        if self.global_pool is not None:
            ks = x.shape[-1]
            x = self.global_pool(x, ks)
            # x = self.global_pool(x)
            if self.dbg_mode:
                x = self.glob_pool_print(x)
            if self.compress:
                x = self.conv1x1(x)
                x = self.c1x1_print(x)
        return x


class MotionEncoder(nn.Module):
    def __init__(self, channels, kernel_size=5, dbg_mode=False):
        super(MotionEncoder, self).__init__()

        model = []
        acti = nn.LeakyReLU(0.2)
        nr_layer = len(channels) - 1

        if dbg_mode:
            dbg_header_msg = f'nr_layer:{nr_layer}, channels:{channels}, kernel_size:{kernel_size}'
            model.append(PrintLayer(layer_name=f'General Info', msg=dbg_header_msg, is_header=True, dbg_mode=dbg_mode))

        # model.append(nn.Sigmoid())
        model.append(nn.BatchNorm1d(channels[0]))
        for i in range(nr_layer):
            pad = (kernel_size - 2) // 2
            model.append(nn.ReflectionPad1d(pad))
            model.append(PrintLayer(layer_name=f'ReflectionPad({i})', msg='Output Shape Is', dbg_mode=dbg_mode))
            model.append(nn.Conv1d(channels[i], channels[i + 1],
                                   kernel_size=kernel_size, stride=2))
            model.append(PrintLayer(layer_name=f'Conv1D({i})',
                                    msg=f'Kernel Size={kernel_size}, Stride=2, Output Shape Is', dbg_mode=dbg_mode))
            model.append(acti)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class FtNet(nn.Module):
    """
    This is the ensembled net object - meaning it combines the different part of the network.
    """

    def __init__(self, mot_en_channels, body_en_channels, cls_head_dims, trj_head_dims,
                 global_pool=None, convpool=None, compress=False, dbg_mode=False):
        super(FtNet, self).__init__()

        self.dbg_mode = dbg_mode

        # self.data_bn = nn.BatchNorm1d(mot_en_channels[0])
        self.mot_encoder = MotionEncoder(mot_en_channels, kernel_size=5, dbg_mode=dbg_mode)
        self.static_encoder = StaticEncoder(body_en_channels, kernel_size=5, global_pool=global_pool, convpool=convpool,
                                            compress=compress, dbg_mode=dbg_mode)

        # self.cls_head_dims = cls_head_dims
        self.cls_head = Head(dims=cls_head_dims, dbg_mode=dbg_mode)
        if trj_head_dims is not None:
            self.trj_head = Head(dims=trj_head_dims, dbg_mode=dbg_mode)
        # for name, param in self.mot_encoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        # self.mot_encoder.apply(init_weights)
        # for name, param in self.mot_encoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        # self.static_encoder.apply(init_weights)
        # self.mot_pooling = nn.AdaptiveAvgPool2d((None, 4))

        # self.fc1 = nn.Linear(768, 192)
        # self.fc2 = nn.Linear(192, 48)
        # self.fc3 = nn.Linear(48, 2)

        self.mot_encoder_print = PrintLayer(layer_name='Motion Encoder', msg='Output Shape Is', dbg_mode=dbg_mode)
        # self.mot_pooling_print = PrintLayer(layer_name='Motion Encoder Pooling', msg='Output Shape Is', dbg_mode=dbg_mode)
        self.static_encoder_print = PrintLayer(layer_name='Static Encoder', msg='Output Shape Is', dbg_mode=dbg_mode)
        # self.fc1_print = PrintLayer(layer_name='Fully Connected 1', msg='Output Shape Is', dbg_mode=dbg_mode)
        # self.fc2_print = PrintLayer(layer_name='Fully Connected 2', msg='Output Shape Is', dbg_mode=dbg_mode)
        # self.fc3_print = PrintLayer(layer_name='Fully Connected 3', msg='Output Shape Is', dbg_mode=dbg_mode)
        # self.lin_act = F.relu

    def forward(self, x):
        # x = self.data_bn(x)
        mot_out = self.mot_encoder(x)
        mot_out = self.mot_encoder_print(mot_out)

        stat_out = self.static_encoder(x)
        stat_out = self.static_encoder_print(stat_out)

        stat_out = stat_out.repeat(1, 1, mot_out.shape[-1])
        feat_map = torch.cat([mot_out, stat_out], dim=1)
        feat_map = feat_map.view(-1, 768)

        cls_out = self.cls_head(feat_map)
        trj_out = None
        if hasattr(self, 'trj_head'):
            trj_out = self.trj_head(feat_map)
        return cls_out, trj_out

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            begin_name = name.split('.')[0]
            if name not in own_state or own_state[name].shape != param.data.shape or begin_name == 'cls_head':
                print(name)
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            own_state[name].copy_(param)

    def print_weight_by_name(self, name):
        own_state = self.state_dict()
        # print(own_state)
        print(own_state[name][0][0])
