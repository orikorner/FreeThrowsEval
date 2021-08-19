# from agent.base_agent import BaseAgent
# from functional.motion import get_foot_vel
from utils.utils import TrainClock
import torch
import torch.nn as nn
import torch.optim as optim
import os.path as osp
from abc import abstractmethod


class Moderator(object):
    def __init__(self, config, net):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.net = net
        self.clock = TrainClock()
        self.device = config.device

        # set optimizer
        self.optimizer = None

        # Scheduler - EXP LR
        self.scheduler = None
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

    @abstractmethod
    def forward(self, data):
        pass

    def save_network(self, name=None):
        if name is None:
            save_path = osp.join(self.model_dir, "model_epoch{}.pth".format(self.clock.epoch))
        else:
            save_path = osp.join(self.model_dir, name)
        torch.save(self.net.cpu().state_dict(), save_path)
        self.net.to(self.device)

    def pre_train_load_network(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        self.net.load_state_dict(state_dict)

    def load_network(self, epoch):
        load_path = osp.join(self.model_dir, "model_epoch{}.pth".format(epoch))
        state_dict = torch.load(load_path)
        self.net.load_state_dict(state_dict)

    @abstractmethod
    def update_network(self, loss_dcit, info=None):
        pass
        # loss = sum(loss_dcit.values())
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

    def update_learning_rate(self, metrics):
        self.scheduler.step()
        # self.scheduler.step(metrics)

    def train_func(self, data):
        self.net.train()

        outputs, losses = self.forward(data)

        self.update_network(losses, data['objective'][0])

        return outputs, losses

    def val_func(self, data):
        self.net.eval()

        with torch.no_grad():
            outputs, losses = self.forward(data)
        # print(outputs[0])
        # print(outputs['cls'][0])
        return outputs, losses