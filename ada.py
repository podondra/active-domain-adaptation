import math
from random import randint

import click
import numpy as np
import pandas as pd
from scipy.stats import norm
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
EPS = 1e-6
MODELPATH = "models/{}.pt"
ENSEMBLEPATH = "models/{}-{}.pt"
SPLIT_COLUMN = "wtd_std_Valence"
TARGET_COLUMN = "critical_temp"


def f(x):
    return x + 0.3 * torch.sin(2 * math.pi * x)


def get_forward(n=1000):
    torch.manual_seed(13)
    x = torch.empty(n, 1).uniform_(0.0, 1.0)
    y = f(x) + torch.empty(n, 1).normal_(std=0.1)
    return None, x, y


def get_inverse(n=1000):
    _, x, y = get_forward(n)
    return None, y, x


def get_protein(datapath="data/CASP.csv"):
    df = pd.read_csv(datapath, dtype=np.float32)
    return df, df.values[:, 1:], df.values[:, :1]


def get_superconductivity(datapath="data/superconduct/train.csv"):
    """The superconductivity data set is available on-line,
    at https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data.
    See Hamidieh (2018) for more information.
    The goal is to predict the critical temperature."""
    df = pd.read_csv(datapath)
    X = df.drop([TARGET_COLUMN, SPLIT_COLUMN], axis=1).values
    y = df[[TARGET_COLUMN]].values
    return df, X, y


def get_wine(datapath="data/winequality-red.csv"):
    df = pd.read_csv(datapath, dtype=np.float32, sep=";")
    return df, df.values[:, :11], df.values[:, 11:]


# TODO You should respect the following train / test split:
# train: first 463,715 examples
# test: last 51,630 examples
# It avoids the 'producer effect' by making sure no song
# from a given artist ends up in both the train and test set.
def get_year(datapath="data/YearPredictionMSD.txt"):
    df = pd.read_csv(datapath, dtype=np.float32, header=None)
    return df, df.values[:, 1:], df.values[:, :1]


DATASETS = {
        "forward": get_forward,
        "inverse": get_inverse,
        "protein": get_protein,
        "wine": get_wine,
        "superconductivity": get_superconductivity,
        "year": get_year}


def domain_split(df, X, y, percentile_low, percentile_up):
    """Domain split inspired by Mathelin et al. (2022) and Pardoe & Stone (2010)."""
    index = (percentile_low < df[SPLIT_COLUMN]) & (df[SPLIT_COLUMN] <= percentile_up)
    return X[index], y[index]


def test_split(X, y, test_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return (X_train, X_test), (y_train, y_test)


def data_preparation(arrays):
    scaler = StandardScaler().fit(arrays[0])   # fit on (source) train
    return map(scaler.transform, arrays), scaler


def numpy2torch(array):
    return torch.from_numpy(array.astype(np.float32)).to(DEVICE)

def get_dataset(dataname, seed, validation):
    df, X, y = DATASETS[dataname]()
    Xs, ys = test_split(X, y, 0.1, seed)
    if validation:    # split the training set and return validation set (not test set)
        Xs, ys = test_split(Xs[0], ys[0], 0.2, seed)
    Xs, X_scaler = data_preparation(Xs)
    ys, y_scaler = data_preparation(ys)
    datasets = map(TensorDataset, map(numpy2torch, Xs), map(numpy2torch, ys))
    return tuple(datasets), (X_scaler, y_scaler)


def get_datasets():
    df, X, y = get_superconductivity()
    percentiles = df[SPLIT_COLUMN].quantile([0.25, 0.75])
    domain_source = domain_split(df, X, y, -np.inf, percentiles[0.25])
    domain_target = domain_split(df, X, y, percentiles[0.75], np.inf)
    Xs_source, ys_source = test_split(*domain_source, 0.2, seed=60)
    Xs_target, ys_target = test_split(*domain_target, 0.2, seed=96)
    Xs, _ = data_preparation(Xs_source + Xs_target)    # "+" chains tuples
    ys, y_scaler = data_preparation(ys_source + ys_target)
    datasets = map(TensorDataset, map(numpy2torch, Xs), map(numpy2torch, ys))
    return tuple(datasets), y_scaler


def get_y(dataset):
    return torch.concat([batch[-1] for batch in DataLoader(dataset, batch_size=BS)]).cpu()


def pit(mean, var, y):
    var = var.clamp(min=EPS)
    return norm.cdf(y, loc=mean, scale=torch.sqrt(var))


def transform(mean, var, y_scaler):
    mean = torch.from_numpy(y_scaler.inverse_transform(mean))
    var = torch.from_numpy(y_scaler.var_) * var
    return mean, var


def log(model, trainset, testset, y_scaler):
    wandb.log({
        "train": model.metrics(model.predict(trainset), get_y(trainset), y_scaler),
        "test": model.metrics(model.predict(testset), get_y(testset), y_scaler)})


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def loss(self, y_pred, y):
        return torch.mean(self.loss_function(*y_pred, y))

    def train_epoch(self, trainloader, optimiser):
        self.train()
        for X_batch, y_batch in trainloader:
            optimiser.zero_grad()
            self.loss(self(X_batch), y_batch).backward()
            optimiser.step()
        return self

    def train_epochs(self, trainset, testset, hyperparams, y_scaler):
        optimiser = Adam(self.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
        scheduler = StepLR(optimiser, step_size=hyperparams["step"], gamma=hyperparams["gamma"])
        trainloader = DataLoader(trainset, batch_size=hyperparams["bs"], shuffle=True)
        log(self, trainset, testset, y_scaler)
        for epoch in range(1, hyperparams["epochs"] + 1):
            self.train_epoch(trainloader, optimiser)
            scheduler.step()
            log(self, trainset, testset, y_scaler)
        return self

    def load(self, modelname):
        self.load_state_dict(torch.load(MODELPATH.format(modelname)))

    def save(self, modelname):
        torch.save(self.state_dict(), MODELPATH.format(modelname))


def nll(mean, var, y):
    return F.gaussian_nll_loss(mean, y, var, eps=EPS, full=True, reduction="none")


def crps(mean, var, y):
    # clamp for stability: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gaussian_nll_loss
    var = var.clone()  # not to modify the original var
    with torch.no_grad():
        var.clamp_(min=EPS)
    std_pred = torch.sqrt(var)
    y_std = (y - mean) / std_pred
    cdf = 0.5 + 0.5 * torch.erf(y_std / math.sqrt(2.0))
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * y_std ** 2)
    return std_pred * (y_std * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def se(mean, var, y):
    return (y - mean) ** 2


class Network(Model):
    def __init__(self, features, hyperparams):
        super().__init__()
        self.loss_function = {"crps": crps, "nll": nll, "se": se}[hyperparams["loss"]]
        neurons = hyperparams["neurons"]
        self.model = nn.Sequential(
            nn.Linear(features, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 1))
        self.model_var = nn.Sequential(
            nn.Linear(features, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 1))
        self.to(DEVICE)

    def forward(self, X):
        # TODO torch.exp versus F.softplus output function for variance
        mean = self.model(X)
        var = torch.exp(self.model_var(X))
        return mean, var

    @torch.no_grad()
    def predict(self, dataset):
        self.eval()
        output = [self(batch[0]) for batch in DataLoader(dataset, batch_size=BS)]
        mean, var = tuple(zip(*output))
        return torch.concat(mean).cpu(), torch.concat(var).cpu()

    def metrics(self, y_pred, y, y_scaler):
        mean, var = y_pred
        mean, var = transform(mean, var, y_scaler)
        y = torch.from_numpy(y_scaler.inverse_transform(y))
        return {
                "crps": torch.mean(crps(mean, var, y)).item(),
                "mae": torch.mean(torch.abs(mean - y)).item(),
                "nll": torch.mean(nll(mean, var, y)).item(),
                "pit": wandb.Histogram(pit(mean, var, y), num_bins=32),
                "rmse": torch.sqrt(torch.mean(torch.square(mean - y))).item(),
                "sharpness": torch.mean(var).item(),
                "var": wandb.Histogram(var)}


def gmm_crps(coeffs, means, variances, y):
    ...


def gmm_nll(coeffs, means, variances, y):
    variances = variances.clone()
    with torch.no_grad():
        variances.clamp_(min=EPS)
    return -torch.logsumexp(torch.log(coeffs) - 0.5 * torch.log(2.0 * math.pi * variances) - 0.5 * ((y - means) ** 2 / variances), dim=-1)


class MixtureDensityNetwork(Model):
    def __init__(self, features, hyperparams):
        super().__init__()
        self.loss_function = {"crps": gmm_crps, "nll": gmm_nll}[hyperparams["loss"]]
        self.K = hyperparams["k"]
        neurons = hyperparams["neurons"]
        self.model = nn.Sequential(
            nn.Linear(features, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 3 * self.K))
        self.to(DEVICE)

    def forward(self, X):
        output = self.model(X)
        coeffs = F.softmax(output[:, :self.K], dim=-1)
        means = output[:, self.K:(2 * self.K)]
        variances = torch.exp(output[:, (2 * self.K):])
        return coeffs, means, variances

    @torch.no_grad()
    def predict(self, dataset):
        self.eval()
        output = [self(batch[0]) for batch in DataLoader(dataset, batch_size=BS)]
        coeffs, means, variances = tuple(zip(*output))
        return torch.cat(coeffs).cpu(), torch.cat(means).cpu(), torch.cat(variances).cpu()

    def metrics(self, y_pred, y, y_scaler):
        return {"nll": self.loss(y_pred, y).item()}


class DeepEnsemble:
    def __init__(self, features, hyperparams):
        self.models = [Network(features, hyperparams) for _ in range(hyperparams["m"])]

    @torch.no_grad()
    def predict(self, dataset):
        outputs = [model.predict(dataset) for model in self.models]
        means, variances = tuple(zip(*outputs))
        means = torch.concat(means, dim=1)
        variances = torch.concat(variances, dim=1)
        mean = torch.mean(means, dim=1, keepdim=True)
        var = torch.mean(variances + torch.square(means), dim=1, keepdim=True) - torch.square(mean)
        return mean, var

    def train_epochs(self, trainset, testset, hyperparams, y_scaler):
        optimisers = [
                Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
                for model in self.models]
        schedulers = [StepLR(optimiser, step_size=hyperparams["step"], gamma=hyperparams["gamma"])
                for optimiser in optimisers]
        trainloader = DataLoader(trainset, batch_size=hyperparams["bs"], shuffle=True)
        log(self, trainset, testset, y_scaler)
        for epoch in range(1, hyperparams["epochs"] + 1):
            for model, optimiser, scheduler in zip(self.models, optimisers, schedulers):
                model.train_epoch(trainloader, optimiser)
                scheduler.step()
            log(self, trainset, testset, y_scaler)
        return self

    def load(self, modelname):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(ENSEMBLEPATH.format(modelname, i)))

    def save(self, modelname):
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), ENSEMBLEPATH.format(modelname, i))


@click.command()
@click.option("--bs", default=64, help="Batch size.")
@click.option("--dataname", required=True, help="Data set name.")
@click.option("--epochs", default=1000, help="Number of epochs.")
@click.option("--gamma", default=1.0, help="Multiplicative factor of learning rate decay.")
@click.option("--k", default=1, help="Number of MDN components")
@click.option("--loss", required=True, help="Loss function.")
@click.option("--lr", default=0.001, help="Learning rate.")
@click.option("--neurons", default=50, help="Number of neurons in the first hidden layer.")
@click.option("--m", default=1, help="Number of model in deep ensemble.")
@click.option("--modelname", type=click.Path(exists=True, dir_okay=False))
@click.option("--seed", default=50, help="Seed for train and test set split.")
@click.option("--step", default=1, help="Period of learning rate decay.")
@click.option("--validation", is_flag=True)
@click.option("--wd", default=0.0, help="Weight decay.")
def train(**hyperparams):
    # reproducibility
    torch.manual_seed(53)
    torch.backends.cudnn.benchmark = False
    # TODO domain adaption setting
    # datasets, y_scaler = get_datasets()
    # trainset_source, testset_source, _, _ = datasets
    (trainset, testset), (_, y_scaler) = get_dataset(
            hyperparams["dataname"], hyperparams["seed"], hyperparams["validation"])
    features = trainset.tensors[0].shape[1]
    # generate random run name with loss information
    runname = "{}-{}-{}-{}".format(
            hyperparams["loss"], hyperparams["m"], hyperparams["k"], randint(1000, 9999))
    with wandb.init(config=hyperparams, name=runname, project="uci"):
        config = wandb.config
        if config["m"] > 1:
            model = DeepEnsemble(features, config)
        elif config["k"] > 1:
            model = MixtureDensityNetwork(features, config)
        else:
            model = Network(features, config)
        if config["modelname"] is not None:
            model.load(config["modelname"])
        model.train_epochs(trainset, testset, config, y_scaler)
        model.save(runname)


if __name__ == "__main__":
    train()
