# -*- coding: utf-8 -*-

import torch
import numpy as np

class PitchFlip(object):
    """
    Horizontal flip a given matrix randomly with a given probability
    """

    def __init__(self, p=1.):
        self.p = p
        
    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be flipped.
        Returns:
            Numpy array: Randomly flipped matrix.
        """
        return torch.flip(data, [0])
    
    def __repr__(self):
        return self.__class__.__name__
    
class TimeFlip(object):
    """
    Horizontal flip a given matrix randomly with a given probability
    """

    def __init__(self, p=1.):
        self.p = p
        
    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be flipped.
        Returns:
            Numpy array: Randomly flipped matrix.
        """
        return torch.flip(data, [1])
    
    def __repr__(self):
        return self.__class__.__name__
   
class NoiseGaussian(object):
    """
    Adds gaussian noise to a given matrix.
    
    Args:
        factor (int): scale of the Gaussian noise. default: 1e-5
    """

    def __init__(self, factor=1e-5):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor of data
        Returns:
            Tensor: Noisy tensor with additive Gaussian noise
        """
        data = data + (np.random.randn(data.shape[0], data.shape[1]) * self.factor)
        return data;
    
class OutliersZeroRandom(object):
    """
    Randomly add zeroed-out outliers (without structure)
    
    Args:
        factor (int): Percentage of outliers to add. default: .25
    """

    def __init__(self, factor=.25):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor of data
        Returns:
            Tensor: Tensor with randomly zeroed-out outliers
        """
        dataSize = data.size
        tmpData = data.copy();
        # Add random outliers (here similar to dropout mask)
        tmpIDs = np.floor(np.random.rand(int(np.floor(dataSize * self.factor))) * dataSize)
        for i in range(tmpIDs.shape[0]):
            if (tmpIDs[i] < data.size):
                tmpData.ravel()[int(tmpIDs[i])] = 0
        return tmpData

class MaskRows(object):
    """
    Put random rows to zeros
    
    Args:
        factor (int): Percentage to be put to zero. default: .2
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        data = data.clone()
        tmpIDs = np.floor(np.random.rand(int(np.floor(data.shape[0] * self.factor))) * (data.shape[0]))
        for i in range(tmpIDs.shape[0]):
            if tmpIDs[i] < data.shape[0]:
                data[int(tmpIDs[i]), :] = 0
        return data
    
    def __repr__(self):
        return self.__class__.__name__

class MaskColumns(object):
    """
    Put random columns to zeros
    
    Args:
        factor (int): Percentage to be put to zero. default: .2
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        data = data.clone()
        tmpIDs = np.floor(np.random.rand(int(np.floor(data.shape[1] * self.factor))) * (data.shape[1]))
        for i in range(tmpIDs.shape[0]):
            if tmpIDs[i] < data.shape[1]:
                data[:, int(tmpIDs[i])] = 0
        return data
    
    def __repr__(self):
        return self.__class__.__name__

class Transpose(object):
    """
    Put random columns to zeros
    
    Args:
        factor (int): Percentage to be put to zero. default: .2
    """

    def __init__(self, value=11):
        self.value = value

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        cur_tr = np.random.randint(1, self.value)
        cur_sign = np.random.randint(0, 2)
        data_tr = torch.zeros_like(data)
        if (cur_sign):
            data_tr[:-cur_tr,:] = data[cur_tr:,:]
        else:
            data_tr[cur_tr:,:] = data[:-cur_tr,:]
        return data_tr
    
    def __repr__(self):
        return self.__class__.__name__