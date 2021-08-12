from moderator.base_moderator import Moderator
# from functional.motion import get_foot_vel
from utils.utils import TrainClock
import torch
import torch.nn as nn
import torch.optim as optim
import os.path as osp


class TrajectoryModerator(Moderator):
    def __init__(self, config, net, lr):
        super(TrajectoryModerator, self).__init__(config, net)

        # self.log_dir = config.log_dir
        # self.model_dir = config.model_dir
        # self.net = net
        # self.clock = TrainClock()
        # self.device = config.device

        # set loss function
        # self.loss = nn.MSELoss()
        self.loss = nn.L1Loss()
        # self.loss = nn.BCELoss()
        # self.loss = nn.SmoothL1Loss()
        # self.loss = nn.CrossEntropyLoss() # TODO: .to(self.device)
        # set optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr)

        # Scheduler - EXP LR
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        # Scheduler - Multi Step
        # milestones = [149]
        # milestones = [85, 120]
        milestones = [200]
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        # Scheduler - Cyclic
        # self.optimizer = optim.Adam(self.net.parameters(), 0.00001)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.00001, max_lr=0.0001, step_size_up=5,
        #                                                    mode="triangular", cycle_momentum=False)
        # Scheduler - Plat
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, verbose=True)

    def forward(self, data):
        inputs = data['motion'].to(self.device)
        labels = data['shot_traj_coeffs'].to(self.device)
        # update loss metric
        losses = {}

        outputs = self.net(inputs)

        losses['mse'] = self.loss(outputs, labels)

        return outputs, losses


class ClassifierModerator(Moderator):
    def __init__(self, config, net, lr):
        super(ClassifierModerator, self).__init__(config, net)

        # self.log_dir = config.log_dir
        # self.model_dir = config.model_dir
        # self.net = net
        # self.clock = TrainClock()
        # self.device = config.device

        # set loss function
        # self.mse = nn.MSELoss()
        self.loss = nn.CrossEntropyLoss() # TODO: .to(self.device)
        # set optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr)

        # Scheduler - EXP LR
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        # Scheduler - Multi Step
        # milestones = [149]
        # milestones = [85, 120]
        # milestones = [40, 90]
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        # Scheduler - Cyclic
        # self.optimizer = optim.Adam(self.net.parameters(), 0.00001)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.00001, max_lr=0.0001, step_size_up=5,
        #                                                    mode="triangular", cycle_momentum=False)
        # Scheduler - Plat
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, verbose=True)

    def forward(self, data):
        inputs = data['motion'].to(self.device)
        labels = data['label'].to(self.device).reshape(-1)
        # update loss metric
        losses = {}

        outputs = self.net(inputs)

        losses['cross_entropy'] = self.loss(outputs, labels)

        return outputs, losses
