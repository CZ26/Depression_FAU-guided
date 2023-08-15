import sys
import random
import logging

import numpy as np
import torch
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logger(level=logging.INFO):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def ccc_score(x, y):
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    covariance = np.nanmean((x - x_mean) * (y - y_mean))
    x_var = np.nanmean((x - x_mean) ** 2)
    y_var = np.nanmean((y - y_mean) ** 2)
    CCC = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)
    return CCC

def mse(x,y):
    return np.sqrt(mean_squared_error(x, y))
def rmse(x,y):
    return mean_absolute_error(x, y)