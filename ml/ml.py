import numpy as np
from scipy.stats import linregress


def momentum(values):
    returns = np.log(values)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = linregress(x, returns)
    annualized = (1 + slope) ** 252
    return annualized * (rvalue ** 2)


def momentum2(values):
    returns = np.log(values)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = linregress(x, returns)
    return ((slope * 2) * (rvalue ** 2)) * 100