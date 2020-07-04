import numpy as np
import torch


def explained_variance(y_pred, y):
    """
    Taken from: https://github.com/openai/baselines

    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """ 
    var_y = torch.var(y)
    return np.nan if var_y==0 else 1 - torch.var(y-y_pred)/var_y


def safe_mean(xs):
    """
    Taken from: https://github.com/openai/baselines

    Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
    """
    return np.nan if len(xs) == 0 else np.mean(xs)


def safe_divide(top, bottom):
    """
    Avoid division error.
    """
    return np.nan if bottom == 0 else top/bottom   
