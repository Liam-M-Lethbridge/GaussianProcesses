"""Contains functions used for generating synthetic data with various patterns."""

import numpy as np

def generateLinearData(x: np.ndarray, m: float = 1, c: float = 0, sigma: float = 1):
    """generates a linear map according to the function y=mx+c + s. s is generated from a normal distribution N(0,sigma)
    Args:
        x: the list of x coords.
        m: gradient coefficient.
        c: constant.
        sigma: the standard deviation of the normal distribution sampled from.
    """

    s = np.random.normal(0,sigma,len(x))

    return x*m+c+s



def generateSinData(x: np.ndarray, period: float = 10,  sigma: float = 1):
    """generates a linear map according to the function y=mx+c + s. s is generated from a normal distribution N(0,sigma)
    Args:
        x: the list of x coords.
        period: the length of the sin wave.
        sigma: the standard deviation of the normal distribution sampled from.
    """

    s = np.random.normal(0,sigma,len(x))

    return np.sin(x/period*2*np.pi)+s


