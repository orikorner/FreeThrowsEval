from dataset import get_dataloader
from common import config
from model import get_network
from utils.utils import cycle
from moderator import get_training_moderator
from utils.visualization import plot_classes_preds, get_predictions
import numpy as np
import torch
import os
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

torch.backends.cudnn.benchmark = True

"""
CMD: train.py -g 0
"""
from torch.utils.tensorboard.summary import hparams


class CorrectedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='skeleton', help='exp name')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to weights')
    parser.add_argument('--obj-mode', type=str, default='cls', help='classification (cls) or Trajectory (trj) training')
    # parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('--poly-deg', type=int, default=2, help="ball trajectory polynomial degree")
    parser.add_argument('--n-epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    parser.add_argument('-aug', action='store_true', default=False, help="specify augmentations")
    parser.add_argument('--dbg-mode', action='store_true', default=False, help="Prints model info")
    parser.add_argument('--tsb-emb', action='store_true', default=False, help="With Tensorboard Embedding")
    parser.add_argument('--w-extras', action='store_true', default=False, help="With 2 dataloaders")

    args = parser.parse_args()
    return args


def get_v_tensor(w, h):
    out = np.zeros((w, h))
    out[:, h//3:(2*h)//3] = 255
    return out


def get_x_tensor(w, h):
    out = np.zeros((w, h))
    out[w//3:(w*2)//3, :] = 255
    return out


def get_val_dummy_imgs(n_val, w, h, labels):
    final_mat = np.zeros((n_val, w, h))
    for i in range(n_val):
        if labels[i] == 0:
            final_mat[i, :, :] = get_v_tensor(w, h)
        else:
            final_mat[i, :, :] = get_x_tensor(w, h)

    final_mat = torch.from_numpy(final_mat)
    final_mat.unsqueeze_(1)
    return final_mat


def main():
    args = parse_args()

    config.initialize(args)

    if config.objective_mode == 'trj':
        batches = [16]
        lr_list = [0.0001]
        trj_weights = [0.25]
        # gaus_noises = [None]
        gaus_noises = [0.0005]
        # once = True
        for curr_batch in batches:
            for curr_lr in lr_list:
                for curr_trj_weight in trj_weights:
                    for curr_gaus_noise in gaus_noises:
                        print(f'==================== {curr_batch} - {curr_lr} - {curr_trj_weight} - {curr_gaus_noise} ===============================')

                        net = get_network(config)
                        if args.checkpoint is not None:
                            net.load_state_dict(torch.load(args.checkpoint))
                        net = net.to(config.device)

                        tr_moder = get_training_moderator(config, net, curr_lr, curr_trj_weight)
                        clock = tr_moder.clock

                        train_tb = CorrectedSummaryWriter(os.path.join(config.log_dir, f'Train_GN_{curr_gaus_noise}_LR_{curr_lr}_TRJW_{curr_trj_weight}'))
                        val_tb = CorrectedSummaryWriter(os.path.join(config.log_dir, f'Val_GN_{curr_gaus_noise}_LR_{curr_lr}_TRJW_{curr_trj_weight}'))

                        train_loader = get_dataloader('train', config, curr_batch, config.num_workers, gauss_noise=curr_gaus_noise)
                        val_loader = get_dataloader('test', config, config.val_set_len, config.num_workers, shuffle=False)
                        val_loader = cycle(val_loader)
                        train_loader_full = get_dataloader('train_full', config, config.train_set_len, config.num_workers, shuffle=False)
                        train_loader_full = cycle(train_loader_full)
                        # start training
                        for e in range(config.nr_epochs):
                            # begin iteration
                            # running_losses = []
                            pbar = tqdm(train_loader)
                            for b, data in enumerate(pbar):
                                # if once:
                                    # train_tb.add_graph(net, data['motion'].to(config.device))
                                    # train_tb.add_graph(net, torch.ones((1, 30, 45)).to(config.device))
                                    # once = False
                                # train step
                                outputs, losses = tr_moder.train_func(data)

                                losses_values = {k: v.item() for k, v in losses.items()}
                                # record loss to tensorboard - key is loss type
                                for k, v in losses_values.items():
                                    # running_losses.append(v)
                                    train_tb.add_scalar(k, v, clock.step)

                                pbar.set_description("EPOCH[{}][{}/{}]".format(e, b + 1, len(train_loader)))
                                # pbar.set_postfix(OrderedDict({"Loss": sum(losses_values.values())}))
                                pbar.set_postfix(OrderedDict({"Trj Loss": losses_values['trj']}))
                                # pbar.set_postfix(OrderedDict({"Cls Loss": losses_values['cls']}))

                                clock.tick()

                            # Getting Validation Loss and Accuracy (over entire validation set)
                            data = next(val_loader)
                            outputs, losses = tr_moder.val_func(data)

                            losses_values = {k: v.item() for k, v in losses.items()}
                            val_cls_loss = losses_values['cls']
                            val_trj_loss = losses_values['trj']
                            labels = data['cls_labels']
                            w_print = True if (e == config.nr_epochs - 1) else False
                            predictions, _, correct = get_predictions(outputs['cls'], data['name'], labels, w_print=True)
                            val_acc = correct / len(labels)

                            # Getting Train Loss and Accuracy (over entire training set)
                            data = next(train_loader_full)
                            outputs, losses = tr_moder.val_func(data)

                            losses_values = {k: v.item() for k, v in losses.items()}
                            train_cls_loss = losses_values['cls']
                            train_trj_loss = losses_values['trj']

                            labels = data['cls_labels']
                            predictions, _, correct = get_predictions(outputs['cls'], data['name'], labels, w_print=False)
                            train_acc = correct / len(labels)

                            val_tb.add_scalars('Accuracies',
                                               {'Train Acc': train_acc,
                                                'Val Acc': val_acc},
                                               global_step=clock.step)
                            val_tb.add_scalars('Trajectory Loss',
                                               {'Train Loss': train_trj_loss,
                                                'Val Loss': val_trj_loss},
                                               global_step=e)
                            val_tb.add_scalars('Classification Loss',
                                               {'Train Loss': train_cls_loss,
                                                'Val Loss': val_cls_loss},
                                               global_step=e)

                            train_tb.add_scalar('learning_rate', tr_moder.optimizer.param_groups[-1]['lr'], clock.epoch)
                            tr_moder.update_learning_rate(e)

                            if clock.epoch % config.save_frequency == 0:
                                tr_moder.save_network()
                            tr_moder.save_network('latest.pth.tar')
                            clock.tock()

                        train_tb.flush()
                        val_tb.flush()
                        train_tb.close()
                        val_tb.close()
    elif config.objective_mode == 'cls':
        label_img = None
        once = True
        batches = [16]
        lr_list = [0.0001]
        classes = ['X', 'V']
        for curr_batch in batches:
            for curr_lr in lr_list:
                print(f'==================== {curr_batch} - {curr_lr} ===============================')

                net = get_network(config)
                if args.checkpoint is not None:
                    net.load_my_state_dict(torch.load(args.checkpoint))
                net = net.to(config.device)
                # create tensorboard writer
                train_tb = CorrectedSummaryWriter(os.path.join(config.log_dir, f'Train_BSize_{curr_batch}_LR_{curr_lr}'))
                val_tb = CorrectedSummaryWriter(os.path.join(config.log_dir, f'Val_BSize_{curr_batch}_LR_{curr_lr}'))

                # create dataloader
                train_loader = get_dataloader('train', config, curr_batch, config.num_workers)
                train_loader_full = get_dataloader('train', config, config.train_set_len, config.num_workers, shuffle=False)
                train_loader_full = cycle(train_loader_full)
                val_loader = get_dataloader('test', config, config.val_set_len, config.num_workers, shuffle=False)
                val_loader = cycle(val_loader)

                # create training agent
                tr_moder = get_training_moderator(config, net, curr_lr)
                clock = tr_moder.clock

                # start training
                for e in range(config.nr_epochs):
                    # begin iteration
                    activation = {}

                    def get_activation(name):
                        def hook(model, input, output):
                            activation[name] = output.detach().cpu().numpy()

                        return hook

                    running_losses = []
                    pbar = tqdm(train_loader)
                    for b, data in enumerate(pbar):

                        # if once:
                            # train_tb.add_graph(net, data['motion'].to(config.device))
                            # train_tb.add_graph(net, torch.ones((1, 30, 45)).to(config.device))
                            # once = False

                        if args.tsb_emb:
                            net.mot_encoder.register_forward_hook(get_activation('mot_encoder'))
                            net.static_encoder.register_forward_hook(get_activation('static_encoder'))

                        # train step
                        outputs, losses = tr_moder.train_func(data)

                        losses_values = {k: v.item() for k, v in losses.items()}

                        # record loss to tensorboard - key is loss type
                        for k, v in losses_values.items():
                            # running_losses.append(v)
                            train_tb.add_scalar(k, v, clock.step)

                        # labels = data['label']
                        # predictions, _, correct = get_predictions(outputs, data['name'], labels, w_print=False)

                        pbar.set_description("EPOCH[{}][{}/{}]".format(e, b, len(train_loader)))
                        pbar.set_postfix(OrderedDict({"Loss": sum(losses_values.values())}))
                        clock.tick()

                    # Getting Validation Loss and Accuracy (over entire validation set)
                    data = next(val_loader)
                    outputs, losses = tr_moder.val_func(data)

                    losses_values = {k: v.item() for k, v in losses.items()}
                    val_loss = losses_values['cls']

                    labels = data['cls_labels']
                    w_print = True if (e == config.nr_epochs - 1) else False
                    predictions, _, correct = get_predictions(outputs, data['name'], labels, w_print=True)
                    val_acc = correct / len(labels)

                    # Getting Train Loss and Accuracy (over entire training set)
                    data = next(train_loader_full)
                    outputs, losses = tr_moder.val_func(data)

                    losses_values = {k: v.item() for k, v in losses.items()}
                    train_loss = losses_values['cls']

                    labels = data['cls_labels']
                    predictions, _, correct = get_predictions(outputs, data['name'], labels, w_print=False)
                    train_acc = correct / len(labels)

                    # Feature Map Embeddings visualization
                    if args.tsb_emb:
                        if once:
                            label_img = get_val_dummy_imgs(config.train_set_len,
                                                           100, 100, labels.reshape(-1).numpy())
                            once = False

                        motion_enc_out = torch.from_numpy(activation['mot_encoder'])
                        static_enc_out = torch.from_numpy(activation['static_encoder'])

                        static_enc_out = static_enc_out.repeat(1, 1, motion_enc_out.shape[-1])
                        feat_map = torch.cat([motion_enc_out, static_enc_out], dim=1)
                        feat_map = feat_map.reshape(config.train_set_len, 768)
                        class_labels = [classes[pred] for pred in predictions]

                        train_tb.add_embedding(feat_map, metadata=class_labels,
                                               label_img=label_img, global_step=clock.step)

                    val_tb.add_scalars('Classification Loss',
                                       {'Train Loss': train_loss,
                                        'Val Loss': val_loss},
                                       global_step=clock.step)
                    val_tb.add_scalars('Accuracies',
                                       {'Train Acc': train_acc,
                                        'Val Acc': val_acc},
                                       global_step=clock.step)

                    train_tb.add_scalar('learning_rate', tr_moder.optimizer.param_groups[-1]['lr'], clock.epoch)
                    # train_tb.add_scalar('cross_entropy', v, clock.step)
                    # mean_loss = sum(running_losses) / len(running_losses) # This is for Plateu LR decay
                    # tr_moder.update_learning_rate(mean_loss)
                    tr_moder.update_learning_rate(e)

                    if clock.epoch % config.save_frequency == 0:
                        tr_moder.save_network()
                    tr_moder.save_network('latest.pth.tar')
                    clock.tock()

                train_tb.flush()
                val_tb.flush()
                train_tb.close()
                val_tb.close()


if __name__ == '__main__':
    main()
