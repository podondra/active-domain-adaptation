from itertools import chain
import math

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb


BS = 2048   # batch size for testing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELPATH = "models/{}.pt"
N_FEATURES = 80
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
    index = (percentile_low < df[SPLIT_COLUMN]) & (df[SPLIT_COLUMN] <= percentile_up)
    return X[index], y[index]


def test_split(X, y, seed):
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
    Xs_source, ys_source = test_split(*domain_source, seed=60)
    Xs_target, ys_target = test_split(*domain_target, seed=96)
    Xs, _ = data_preparation(Xs_source, Xs_target)
    ys, y_scaler = data_preparation(ys_source, ys_target)
    Xs, ys = map(array2gpu_tensor, Xs), map(array2gpu_tensor, ys)
    datasets = map(TensorDataset, Xs, ys)
    return tuple(datasets), y_scaler


def get_y(dataset):
    return torch.concat([batch[-1] for batch in DataLoader(dataset, batch_size=BS)]).cpu()


def nll(mean_pred, var_pred, y):
    return torch.log(var_pred) + torch.square(y - mean_pred) / var_pred


def crps(mean_pred, var_pred, y):
    std_pred = torch.sqrt(var_pred)
    y_std = (y - mean_pred) / std_pred
    pi = torch.tensor(math.pi)
    pdf = (1.0 / torch.sqrt(2.0 * pi)) * torch.exp(-torch.square(y_std) / 2.0)
    cdf = 0.5 + 0.5 * torch.erf(y_std / torch.sqrt(torch.tensor(2.0)))
    return std_pred * (y_std * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / torch.sqrt(pi))


def metrics(mean, var, y):
    return {
            "crps": torch.mean(crps(mean, var, y)).item(),
            "mae": torch.mean(torch.abs(mean - y)).item(),
            "nll": torch.mean(nll(mean, var, y)).item(),
            "rmse": torch.sqrt(torch.mean(torch.square(mean - y))).item(),
            "sharpness": torch.mean(var).item()}


class Model(nn.Module):
    def __init__(self, hyperparams):
        super(Model, self).__init__()
        self.loss_function = {"crps": crps, "nll": nll}[hyperparams["loss"]]
        neurons = hyperparams["neurons"]
        self.model_mean = nn.Sequential(
            nn.Linear(N_FEATURES, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 1))    # mean and variance
        self.model_var = nn.Sequential(
            nn.Linear(N_FEATURES, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 1))    # mean and variance
        self.to(DEVICE)

    def forward(self, X):
        return self.model_mean(X), F.softplus(self.model_var(X)) + 1e-6

    @torch.no_grad()
    def predict(self, dataset):
        self.eval()
        output = [self(batch[0]) for batch in DataLoader(dataset, batch_size=BS)]
        mean, var = tuple(zip(*output))
        return torch.concat(mean).cpu(), torch.concat(var).cpu()

    def loss(self, y_pred, y):
        return torch.mean(self.loss_function(*y_pred, y))

    def train_epoch(self, trainloader, optimiser):
        self.train()
        for X_batch, y_batch in trainloader:
            optimiser.zero_grad()
            self.loss(self(X_batch), y_batch).backward()
            optimiser.step()
        return self

    def train_epochs(self, trainset, testset, hyperparams):
        optimiser = Adam(self.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
        scheduler = StepLR(optimiser, step_size=hyperparams["step"], gamma=hyperparams["gamma"])
        trainloader = DataLoader(trainset, batch_size=hyperparams["bs"], shuffle=True)
        for epoch in range(1, hyperparams["epochs"] + 1):
            self.train_epoch(trainloader, optimiser)
            mean_train, var_train = self.predict(trainset)
            mean_test, var_test = self.predict(testset)
            wandb.log({
                "train": metrics(mean_train, var_train, get_y(trainset)),
                "test": metrics(mean_test, var_test, get_y(testset)),
                "var_train": wandb.Histogram(var_train),
                "var_test": wandb.Histogram(var_test)})
            scheduler.step()
        return self

    def save(self, modelpath):
        torch.save(self.state_dict(), modelpath)

    def load(self, modelpath):
        self.load_state_dict(torch.load(modelpath))
        return self


class DeepEnsemble:
    def __init__(self, hyperparams):
        self.models = [Model(hyperparams) for _ in range(hyperparams["m"])]

    @torch.no_grad()
    def predict(self, dataset):
        outputs = [model.predict(dataset) for model in self.models]
        means, variances = tuple(zip(*outputs))
        means = torch.concat(means, dim=1)
        variances = torch.concat(variances, dim=1)
        mean = torch.mean(means, dim=1, keepdim=True)
        # TODO why is this correct?
        var = torch.mean(variances + torch.square(means), dim=1, keepdim=True) - torch.square(mean)
        return mean, var

    def train_epochs(self, trainset, testset, hyperparams):
        optimisers = [
                Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
                for model in self.models]
        schedulers = [StepLR(optimiser, step_size=hyperparams["step"], gamma=hyperparams["gamma"])
                for optimiser in optimisers]
        trainloader = DataLoader(trainset, batch_size=hyperparams["bs"], shuffle=True)
        for epoch in range(1, hyperparams["epochs"] + 1):
            for model, optimiser, scheduler in zip(self.models, optimisers, schedulers):
                model.train_epoch(trainloader, optimiser)
                mean_train, var_train = self.predict(trainset)
                mean_test, var_test = self.predict(testset)
                wandb.log({
                    "train": metrics(mean_train, var_train, get_y(trainset)),
                    "test": metrics(mean_test, var_test, get_y(testset)),
                    "var_train": wandb.Histogram(var_train),
                    "var_test": wandb.Histogram(var_test)})
                scheduler.step()
        return self


@click.command()
@click.option("--bs", default=128, help="Batch size.")
@click.option("--epochs", default=1024, help="Number of epochs.")
@click.option("--gamma", default=1.0, help="Multiplicative factor of learning rate decay.")
@click.option("--neurons", default=8, help="Number of neurons in the first hidden layer.")
@click.option("--m", default=5, help="Number of model in the ensemble.")
@click.option("--modelpath", type=click.Path(exists=True, dir_okay=False))
@click.option("--loss", required=True, help="Loss function.")
@click.option("--lr", default=0.001, help="Learning rate.")
@click.option("--step", default=256, help="Period of learning rate decay.")
@click.option("--wd", default=0.0, help="Weight decay.")
# TODO implement train model and train ensemble as subcommands
def train(**hyperparams):
    # reproducibility
    torch.manual_seed(53)
    torch.backends.cudnn.benchmark = False
    # get data
    datasets, y_scaler = get_datasets()
    trainset_source, testset_source, trainset_target, testset_target = datasets
    with wandb.init(config=hyperparams):
        config = wandb.config
        # TODO model = Model(config)
        model = DeepEnsemble(config)
        if hyperparams["modelpath"] is not None:
            model.load(hyperparams["modelpath"])
        model.train_epochs(trainset_source, testset_source, config)
        # TODO model.save(MODELPATH.format(wandb.run.name))


if __name__ == "__main__":
    train()
