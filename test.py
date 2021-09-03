from dataset import get_dataloader
from common import config
from model import get_network
from utils.visualization import print_predictions_info
from utils.utils import cycle
import argparse
import pandas as pd
import os.path as osp
from moderator import get_training_moderator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='skeleton', help='exp name')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to weights')
    # parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config.initialize(args)

    net = get_network(config)
    net = net.to(config.device)

    labels_info_df = pd.read_csv(osp.join(config.data_dir, config.labels_file), header=0)
    labels_info_df = labels_info_df.loc[labels_info_df['phase'] == 'test']

    val_loader = get_dataloader('test', config, config.val_set_len, config.num_workers, shuffle=False)
    data = next(cycle(val_loader))

    tr_moder = get_training_moderator(config, net, lr=1)
    tr_moder.pre_train_load_network(args.checkpoint)

    outputs, losses = tr_moder.val_func(data)

    print_predictions_info(outputs, data['cls_labels'], data['name'], labels_info_df)


if __name__ == '__main__':
    main()
