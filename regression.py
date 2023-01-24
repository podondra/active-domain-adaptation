from itertools import chain
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_COLUMN = "wtd_std_Valence"
TARGET_COLUMN = "critical_temp"


def get_superconductivity(datapath="data/superconduct/train.csv"):
    """The superconductivity data set is available on-line,
    at https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data.
    See Hamidieh (2018) for more information.
    The goal is to predict the critical temperature."""
    df = pd.read_csv(datapath)
    X = df.drop([TARGET_COLUMN, SPLIT_COLUMN], axis=1).values
    y = df[[TARGET_COLUMN]].values
    return df, X, y


def domain_split(df, X, y, percentile_low, percentile_up):
    """Domain split inspired by Mathelin et al. (2022) and Pardoe & Stone (2010)."""
    index = (percentile_low <= df[SPLIT_COLUMN]) & (df[SPLIT_COLUMN] < percentile_up)
    return X[index], y[index]


def test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
    return (X_train, X_test), (y_train, y_test)


def data_preparation(arrays_source, arrays_target):
    scaler = StandardScaler().fit(arrays_source[0])   # fit on source train
    return map(scaler.transform, chain(arrays_source, arrays_target)), scaler


def array2gpu_tensor(array):
    return torch.from_numpy(array.astype(np.float32)).to(DEVICE)


def get_datasets():
    df, X, y = get_superconductivity()
    percentiles = df[SPLIT_COLUMN].quantile([0.25, 0.75])
    domain_source = domain_split(df, X, y, -np.inf, percentiles[0.25])
    domain_target = domain_split(df, X, y, percentiles[0.75], np.inf)
    Xs_source, ys_source = test_split(*domain_source)
    Xs_target, ys_target = test_split(*domain_target)
    Xs, _ = data_preparation(Xs_source, Xs_target)
    ys, y_scaler = data_preparation(ys_source, ys_target)
    Xs, ys = map(array2gpu_tensor, Xs), map(array2gpu_tensor, ys)
    datasets = map(TensorDataset, Xs, ys)
    return tuple(datasets), y_scaler
