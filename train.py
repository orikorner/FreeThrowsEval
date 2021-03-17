from dataset import get_dataloader
from common import config
from model import get_network
# from functional.utils import cycle
from utils.general import cycle
# from agent import get_training_agent
from moderator import get_training_moderator
# from functional.visualization import visulize_motion_in_training
import torch
import os
from collections import OrderedDict
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse

torch.backends.cudnn.benchmark = True

"""
CMD: train.py -g 0
"""


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, help='path to poses')
    parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    parser.add_argument('-aug', action='store_true', default=False, help="specify augmentations")
    parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config.initialize(args)

    net = get_network(config)
    net = net.to(config.device)

    # create tensorboard writer
    train_tb = SummaryWriter(os.path.join(config.log_dir, 'train.events'))
    val_tb = SummaryWriter(os.path.join(config.log_dir, 'val.events'))

    # create dataloader
    train_loader = get_dataloader('train', config, config.batch_size, config.num_workers)
    # mean_pose, std_pose = train_loader.dataset.mean_pose, train_loader.dataset.std_pose
    val_loader = get_dataloader('test', config, config.batch_size, config.num_workers)
    val_loader = cycle(val_loader)

    # create training agent
    tr_moder = get_training_moderator(config, net)
    clock = tr_moder.clock

    # start training
    for e in range(config.nr_epochs):

        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = tr_moder.train_func(data)

            losses_values = {k: v.item() for k, v in losses.items()}

            # record loss to tensorboard
            for k, v in losses_values.items():
                train_tb.add_scalar(k, v, clock.step)

            # visualize
            # if args.vis and clock.step % config.visualize_frequency == 0:
            #     imgs = visulize_motion_in_training(outputs, mean_pose, std_pose)
            #     for k, img in imgs.items():
            #         train_tb.add_image(k, torch.from_numpy(img), clock.step)

            pbar.set_description("EPOCH[{}][{}/{}]".format(e, b, len(train_loader)))
            pbar.set_postfix(OrderedDict({"loss": sum(losses_values.values())}))

            # validation step
            if clock.step % config.val_frequency == 0:
                data = next(val_loader)

                outputs, losses = tr_moder.val_func(data)

                losses_values = {k: v.item() for k, v in losses.items()}

                for k, v in losses_values.items():
                    val_tb.add_scalar(k, v, clock.step)

                # if args.vis and clock.step % config.visualize_frequency == 0:
                #     imgs = visulize_motion_in_training(outputs, mean_pose, std_pose)
                #     for k, img in imgs.items():
                #         val_tb.add_image(k, torch.from_numpy(img), clock.step)

            clock.tick()

        train_tb.add_scalar('learning_rate', tr_moder.optimizer.param_groups[-1]['lr'], clock.epoch)
        tr_moder.update_learning_rate()

        if clock.epoch % config.save_frequency == 0:
            tr_moder.save_network()
        tr_moder.save_network('latest.pth.tar')

        clock.tock()


if __name__ == '__main__':
    main()
