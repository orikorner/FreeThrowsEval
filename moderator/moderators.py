# from agent.base_agent import BaseAgent
# from functional.motion import get_foot_vel
from utils.utils import TrainClock
import torch
import torch.nn as nn
import torch.optim as optim
import os.path as osp


class Moderator(object):
    def __init__(self, config, net, lr):
        # self.inputs_name = ['input1', 'input2', 'input12', 'input21']
        # self.targets_name = ['target1', 'target2', 'target12', 'target21']

        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.net = net
        self.clock = TrainClock()
        self.device = config.device
        n_epochs = config.nr_epochs
        # self.use_triplet = config.use_triplet
        # self.use_footvel_loss = config.use_footvel_loss

        # set loss function
        # self.mse = nn.MSELoss()
        # self.tripletloss = nn.TripletMarginLoss(margin=config.triplet_margin)
        # self.triplet_weight = config.triplet_weight
        # self.foot_idx = config.foot_idx
        # self.footvel_loss_weight = config.footvel_loss_weight
        self.cross_ent_loss = nn.CrossEntropyLoss() # TODO: .to(self.device)
        # set optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr)

        # Scheduler - EXP LR
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        # Scheduler - Multi Step
        # milestones = [149]
        milestones = [40, 90]
        # milestones = [85, 120]
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        # Scheduler - Cyclic
        # self.optimizer = optim.Adam(self.net.parameters(), 0.00001)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.00001, max_lr=0.0001, step_size_up=250,
        #                                                    mode="triangular", cycle_momentum=False)
        # Scheduler - Plat
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, verbose=True)

    def forward(self, data):
        inputs = data['motion'].to(self.device)
        labels = data['label'].to(self.device).reshape(-1)#.argmax(1)
        # update loss metric
        losses = {}

        outputs = self.net(inputs)

        losses['cross_entropy'] = self.cross_ent_loss(outputs, labels)

        return outputs, losses
        # if self.use_triplet:
        #     outputs, motionvecs, staticvecs = self.net.cross_with_triplet(*inputs)
        #     losses['m_tpl1'] = self.triplet_weight * self.tripletloss(motionvecs[2], motionvecs[0], motionvecs[1])
        #     losses['m_tpl2'] = self.triplet_weight * self.tripletloss(motionvecs[3], motionvecs[1], motionvecs[0])
        #     losses['b_tpl1'] = self.triplet_weight * self.tripletloss(staticvecs[2], staticvecs[0], staticvecs[1])
        #     losses['b_tpl2'] = self.triplet_weight * self.tripletloss(staticvecs[3], staticvecs[1], staticvecs[0])
        # else:
        #     outputs = self.net.cross(inputs[0], inputs[1])

        # for i, target in enumerate(targets):
        #     losses['rec' + self.targets_name[i][6:]] = self.mse(outputs[i], target)

        # if self.use_footvel_loss:
        #     losses['foot_vel'] = 0
        #     for i, target in enumerate(targets):
        #         losses['foot_vel'] += self.footvel_loss_weight * self.mse(get_foot_vel(outputs[i], self.foot_idx),
        #                                                                   get_foot_vel(target, self.foot_idx))

        # outputs_dict = {
        #     "output1": outputs[0],
        #     "output2": outputs[1],
        #     "output12": outputs[2],
        #     "output21": outputs[3],
        # }
        # return outputs_dict, losses

    def save_network(self, name=None):
        if name is None:
            # save_path = osp.join(self.model_dir, "model_epoch{}.pth".format(self.curr_epoch))
            save_path = osp.join(self.model_dir, "model_epoch{}.pth".format(self.clock.epoch))
        else:
            save_path = osp.join(self.model_dir, name)
        torch.save(self.net.cpu().state_dict(), save_path)
        self.net.to(self.device)

    def load_network(self, epoch):
        load_path = osp.join(self.model_dir, "model_epoch{}.pth".format(epoch))
        state_dict = torch.load(load_path)
        self.net.load_state_dict(state_dict)

    def update_network(self, loss_dcit):
        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self, metrics):
        self.scheduler.step()
        # self.scheduler.step(metrics)

    def train_func(self, data):
        self.net.train()

        outputs, losses = self.forward(data)

        self.update_network(losses)

        return outputs, losses

    def val_func(self, data):
        self.net.eval()

        with torch.no_grad():
            outputs, losses = self.forward(data)

        return outputs, losses