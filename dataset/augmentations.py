import numpy as np
import torch


class NormalizeMotion(object):
    """
    Normalizes motion sample using Mean and Standard Deviation
    """
    def __init__(self, mean_pose, std_pose):
        """
        :param mean_pose: (J, 2)
        :param std_pose: (J, 2)
        """
        self.mean_pose = mean_pose
        self.std_pose = std_pose

    def __call__(self, motion):
        """
        :param motion: (J, 2, T)
        :param mean_pose: (J, 2)
        :param std_pose: (J, 2)
        :return: Normalized motion (J, 2, T)
        """
        return (motion - self.mean_pose[:, :, np.newaxis]) / self.std_pose[:, :, np.newaxis]

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'Mean: {self.mean_pose},'
            f'Variance: {self.std_pose})'
            )
        return repr_str


class Resize(object):
    """
    Resizes given sample
    """

    def __init__(self, scale):
        """
        :param scale: tuple
        """
        self.scale = scale

    def __call__(self, sample):
        """
        Default value for resize is (Joints * 2, T)
        :param sample: input sample (np.array)
        :return: resized sample
        """
        # xs = sample[:, 0, :]
        # ys = sample[:, 1, :]
        # return np.concatenate((xs, ys), axis=0)
        # TODO
        return sample.reshape(self.scale)

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'Scale: {self.scale})'
            )
        return repr_str


class ToTensor(object):
    """
    Converts ndarrays in sample to Tensor
    """

    def __call__(self, sample):
        """
        Default value for resize is (Joints * 2, T)
        :param sample: input sample (np.array)
        :return: sample as 3D Tensor
        """
        return torch.Tensor(sample)


class GaussianNoise(object):
    """
    Introduces Gaussian Noise to sample
    """

    def __init__(self, mean_pose, std_pose):
        """
        :param mean_pose: (J, 2)
        :param std_pose: (J, 2)
        """
        self.mean_pose = mean_pose
        self.std_pose = std_pose

    def __call__(self, sample):
        """
        :param sample: input sample (np.array)
        :return: sample with added gaussian noise
        """
        sigma = np.ones(sample.shape) * self.std_pose
        mean = np.ones(sample.shape) * self.mean_pose
        sample = sample + np.random.normal(size=sample.shape) * sigma + mean
        return sample

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'Mean: {self.mean_pose},'
            f'Sigma: {self.std_pose})'
            )
        return repr_str


class RandomZeroMask(object):
    """
    Zeros out random percentage of indices
    """

    def __init__(self, p):
        """
        :param p: percentage of indices to mask
        """
        self.p = p

    def __call__(self, sample):
        """
        :param sample: input sample (np.array)
        :return: masked sample
        """
        x, y = sample.shape
        n_select = int((self.p * x * y) // 2)
        rand_j = np.random.choice(np.arange(x), n_select, replace=False)
        rand_t = np.random.choice(np.arange(y), n_select, replace=False)

        sample[rand_j, rand_t] = 0.0
        return sample

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'Shape: {self.shape},'
            f'P: {self.p})'
            )
        return repr_str