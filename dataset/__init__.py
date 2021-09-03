from torch.utils.data import DataLoader, ConcatDataset
from dataset.dataset import BBFTSDataset, BBFTSBatchSchedulerSampler
import numpy as np


def get_dataloader(subset, config, batch_size=16, num_workers=4, gauss_noise=None, shuffle=True):

    dataloader = None

    if config.objective_mode == 'trj' and subset == 'train':
        cls_dataset = BBFTSDataset(subset='train', config=config, gauss_noise=gauss_noise)
        trj_dataset = BBFTSDataset(subset='extras', config=config, gauss_noise=gauss_noise)
        concat_dataset = ConcatDataset([cls_dataset, trj_dataset])

        dataloader = DataLoader(dataset=concat_dataset,
                                sampler=BBFTSBatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                worker_init_fn=lambda _: np.random.seed())
    elif config.objective_mode == 'cls':
        cls_dataset = BBFTSDataset(subset=subset, config=config, gauss_noise=gauss_noise)
        dataloader = DataLoader(dataset=cls_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                worker_init_fn=lambda _: np.random.seed())
    else:
        if subset == 'train_full':
            cls_dataset = BBFTSDataset(subset='train', config=config, gauss_noise=gauss_noise)
        else:
            assert subset == 'test'
            cls_dataset = BBFTSDataset(subset='test', config=config, gauss_noise=gauss_noise)

        dataloader = DataLoader(dataset=cls_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                worker_init_fn=lambda _: np.random.seed())

    return dataloader
