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
    # parser.add_argument('--dataset', type=str, help='path to poses')
    parser.add_argument('--name', type=str, default='skeleton', help='exp name')
    parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    parser.add_argument('-aug', action='store_true', default=False, help="specify augmentations")
    parser.add_argument('--dbg-mode', action='store_true', default=False, help="Prints model info")

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

    label_img = None
    once = True
    batches = [24, 32]
    lr_list = [0.0001]
    classes = ['X', 'V']
    for curr_batch in batches:
        for curr_lr in lr_list:
            print(f'==================== {curr_batch} - {curr_lr} ===============================')

            net = get_network(config)
            net = net.to(config.device)
            # create tensorboard writer
            train_tb = CorrectedSummaryWriter(os.path.join(config.log_dir, f'Train_BSize_{curr_batch}_LR_{curr_lr}'))
            val_tb = CorrectedSummaryWriter(os.path.join(config.log_dir, f'Val_BSize_{curr_batch}_LR_{curr_lr}'))

            # create dataloader
            train_loader = get_dataloader('train', config, curr_batch, config.num_workers)
            train_loader_full = get_dataloader('train', config, 276, config.num_workers, shuffle=False)

            val_loader = get_dataloader('test', config, 50, config.num_workers, shuffle=False)
            val_loader = cycle(val_loader)

            # create training agent
            tr_moder = get_training_moderator(config, net, curr_lr)
            clock = tr_moder.clock

            # start training
            for e in range(config.nr_epochs):
                # e_losses = []
                # e_accuracies = []
                # begin iteration
                pbar = tqdm(train_loader)
                for b, data in enumerate(pbar):

                    activation = {}

                    def get_activation(name):
                        def hook(model, input, output):
                            activation[name] = output.detach().cpu().numpy()

                        return hook

                    net.mot_encoder.register_forward_hook(get_activation('mot_encoder'))
                    net.static_encoder.register_forward_hook(get_activation('static_encoder'))
                    # train step
                    outputs, losses = tr_moder.train_func(data)

                    losses_values = {k: v.item() for k, v in losses.items()}

                    # record loss to tensorboard - key is loss type
                    for k, v in losses_values.items():
                        train_tb.add_scalar(k, v, clock.step)

                    # Batch Loss and Accuracy - for hyper params search
                    # e_losses.append(losses_values['cross_entropy'])
                    labels = data['label']
                    predictions, _, correct = get_predictions(outputs,
                                                              data['name'], data['motion'].to(config.device), labels, w_print=False)
                    # batch_train_acc = correct / len(labels)
                    # e_accuracies.append(batch_train_acc)

                    # Feature maps visualization
                    # if b == 1:
                    #     if once:
                        #     print(data['name'])
                    #         label_img = get_val_dummy_imgs(curr_batch, 100, 100, labels.reshape(-1).numpy())
                    #         once = False
                    #     # features = data['motion'].reshape(data['motion'].shape[0], -1)
                    #     motion_enc_out = torch.from_numpy(activation['mot_encoder'])
                    #     static_enc_out = torch.from_numpy(activation['static_encoder'])
                    #
                    #     static_enc_out = static_enc_out.repeat(1, 1, motion_enc_out.shape[-1])
                    #     feat_map = torch.cat([motion_enc_out, static_enc_out], dim=1)
                    #     feat_map = feat_map.view(-1, 768)
                    #
                    #     class_labels = [classes[pred] for pred in predictions]
                    #     train_tb.add_embedding(feat_map, metadata=class_labels, label_img=label_img, global_step=clock.step)
                        # train_tb.add_embedding(feat_map, metadata=class_labels, label_img=data['motion'].unsqueeze(1), global_step=b)

                    pbar.set_description("EPOCH[{}][{}/{}]".format(e, b, len(train_loader)))
                    pbar.set_postfix(OrderedDict({"Loss": sum(losses_values.values())}))

                    # val step-print every val.frequency batches (e.g val_freq = 10, batch size = 8 -> every 80 samples) = every 10 batches
                    if (clock.step + 1) % config.val_frequency == 0:
                        # Getting Validation Loss and Accuracy (over entire validation set)
                        data = next(val_loader)
                        outputs, losses = tr_moder.val_func(data)

                        losses_values = {k: v.item() for k, v in losses.items()}
                        val_loss = losses_values['cross_entropy']

                        # for k, v in losses_values.items():
                        #     val_tb.add_scalar(k, v, clock.step)

                        labels = data['label']
                        predictions, _, correct = get_predictions(outputs,
                                                        data['name'], data['motion'].to(config.device), labels, w_print=True)
                        val_acc = correct / len(labels)
                        # val_tb.add_figure('Predictions vs. Actual',
                        #                   plot_classes_preds(outputs, vid_names, inputs, labels),
                        #                   global_step=clock.step)

                        # Getting Train Loss and Accuracy (over entire training set)
                        data = next(iter(train_loader_full))
                        outputs, losses = tr_moder.val_func(data)

                        losses_values = {k: v.item() for k, v in losses.items()}
                        train_loss = losses_values['cross_entropy']

                        labels = data['label']
                        predictions, _, correct = get_predictions(outputs,
                                                        data['name'], data['motion'].to(config.device), labels, w_print=False)
                        train_acc = correct / len(labels)

                        # Feature Map Embeddings visualization
                        if once:
                            # print(data['name'])
                            label_img = get_val_dummy_imgs(276, 100, 100, labels.reshape(-1).numpy())
                            once = False

                        motion_enc_out = torch.from_numpy(activation['mot_encoder'])
                        static_enc_out = torch.from_numpy(activation['static_encoder'])

                        static_enc_out = static_enc_out.repeat(1, 1, motion_enc_out.shape[-1])
                        feat_map = torch.cat([motion_enc_out, static_enc_out], dim=1)
                        # feat_map = feat_map.view(-1, 768)
                        feat_map = feat_map.reshape(276, 768)
                        class_labels = [classes[pred] for pred in predictions]

                        train_tb.add_embedding(feat_map, metadata=class_labels,
                                               label_img=label_img, global_step=clock.step)
                        val_tb.add_scalars('Losses',
                                           {'Train Loss': train_loss,
                                            'Val Loss': val_loss},
                                           global_step=clock.step)
                        val_tb.add_scalars('Accuracies',
                                           {'Train Acc': train_acc,
                                            'Val Acc': val_acc},
                                           global_step=clock.step)

                    clock.tick()

                # data = next(val_loader)
                # outputs, losses = tr_moder.val_func(data)
                #
                # losses_values = {k: v.item() for k, v in losses.items()}
                # val_loss = losses_values['cross_entropy']
                # labels = data['label']
                #
                # predictions, _, correct = get_predictions(outputs,
                #                                           data['name'], data['motion'].to(config.device), labels,
                #                                           w_print=False)
                # val_acc = correct / len(labels)

                # train_tb.add_hparams({'lr': curr_lr, 'bsize': curr_batch},
                #                      {'train accuracy': sum(e_accuracies)/len(e_accuracies),
                #                       'train loses': sum(e_losses)/len(e_losses),
                #                       'val accuracy': val_acc,
                #                       'val loss': val_loss})
                train_tb.add_scalar('learning_rate', tr_moder.optimizer.param_groups[-1]['lr'], clock.epoch)
                tr_moder.update_learning_rate()

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
