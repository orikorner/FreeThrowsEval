from moderator.base_moderator import Moderator
import torch
import torch.nn as nn
import torch.optim as optim


class TrajectoryModerator(Moderator):
    def __init__(self, config, net, lr, trj_weight=0.5):
        super(TrajectoryModerator, self).__init__(config, net)

        # set loss function
        # self.loss = nn.MSELoss()
        self.trj_weight = trj_weight
        self.trj_loss = nn.L1Loss()
        # self.loss = nn.BCELoss()
        # self.loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss()
        # set optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr)

        # Scheduler - EXP LR
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        # Scheduler - Multi Step
        # milestones = [149]
        # milestones = [200]
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
        cls_labels = data['cls_labels'].to(self.device).reshape(-1)
        trj_labels = data['trj_labels'].to(self.device)

        cls_out, trj_out = self.net(inputs)

        cls_loss = self.cls_loss(cls_out, cls_labels)
        trj_loss = self.trj_loss(trj_out, trj_labels)

        outputs = {'cls': cls_out, 'trj': trj_out}
        losses = {'cls': cls_loss, 'trj': trj_loss}
        return outputs, losses

    def update_network(self, loss_dcit, info=None):
        loss = None
        if info == 'trj':
            loss = self.trj_weight * loss_dcit['trj']
            loss.backward()
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            loss = (1 - self.trj_weight) * loss_dcit['cls']
            loss.backward()


class ClassifierModerator(Moderator):
    def __init__(self, config, net, lr):
        super(ClassifierModerator, self).__init__(config, net)

        # set loss function
        self.loss = nn.CrossEntropyLoss() # TODO: .to(self.device)
        # set optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr)

        # Scheduler - EXP LR
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        # Scheduler - Multi Step
        # milestones = [149]
        # milestones = [85, 120]
        milestones = [40, 90]
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        # Scheduler - Cyclic
        # self.optimizer = optim.Adam(self.net.parameters(), 0.00001)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.00001, max_lr=0.0001, step_size_up=5,
        #                                                    mode="triangular", cycle_momentum=False)
        # Scheduler - Plat
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, verbose=True)

    def forward(self, data):
        inputs = data['motion'].to(self.device)
        labels = data['cls_labels'].to(self.device).reshape(-1)
        # update loss metric
        losses = {}

        outputs, _ = self.net(inputs)

        losses['cls'] = self.loss(outputs, labels)

        return outputs, losses

    # def forward(self, data):
    #     inputs = data['motion'].to(self.device)
    #     labels = data['cls_labels'].to(self.device).reshape(-1)
    #
    #     cls_out, _ = self.net(inputs)
    #     cls_loss = self.cls_loss(cls_out, labels)
    #
    #     outputs = {'cls': cls_out}
    #     losses = {'cls': cls_loss}
    #     return outputs, losses

    def update_network(self, loss_dcit, info=None):
        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print()
        # self.net.print_weight_by_name('cls_head.model.0.weight')
        # print()
