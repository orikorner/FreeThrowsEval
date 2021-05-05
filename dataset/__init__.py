from torch.utils.data import DataLoader
from dataset.dataset import BBFTSDataset
import numpy as np


def get_dataloader(subset, config, batch_size=16, num_workers=4, shuffle=True):

    data = BBFTSDataset(subset, config)

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, worker_init_fn=lambda _: np.random.seed())

    return dataloader
