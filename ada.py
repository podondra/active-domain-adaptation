import math
from random import randint

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb


BINS = 32
BS = 2048   # batch size for testing
CONFIDENCES = torch.linspace(0.0, 1.0, steps=11)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6
MODELPATH = "models/{}.pt"
ENSEMBLEPATH = "models/{}-{}.pt"
SPLIT_COLUMN = "wtd_std_Valence"
TARGET_COLUMN = "critical_temp"


def get_inverse(n=1000, seed=13):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, 1.0, size=(n, 1))
    x = y + 0.3 * np.sin(2 * math.pi * y) + rng.uniform(-0.1, 0.1, size=(n, 1))
    y += rng.normal(scale=0.05, size=(n, 1))
    return None, x, y


def get_protein(datapath="data/CASP.csv"):
    df = pd.read_csv(datapath, dtype=np.float32)
    return df, df.values[:, 1:], df.values[:, :1]


# TODO drop SPLIT_COLUMN elsewhere
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


class StandardScaler:
    def __init__(self):
        pass

    def fit(self, X):
        self.mean = torch.mean(X, dim=0, keepdim=True)
        self.std = torch.std(X, dim=0, unbiased=True, keepdim=True)
        self.variance = self.std ** 2
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return self.std * X + self.mean

    def inverse_transform_variance(self, variance):
        return self.variance * variance


def data_preparation(arrays):
    scaler = StandardScaler().fit(arrays[0])   # fit on (source) train
    return map(scaler.transform, arrays), scaler


def numpy2torch(array):
    return torch.from_numpy(array.astype(np.float32))


def to_device(tensor):
    return tensor.to(DEVICE)


def get_dataset(dataname, seed, validation, preparation):
    df, X, y = DATASETS[dataname]()
    X, y = numpy2torch(X), numpy2torch(y)
    Xs, ys = test_split(X, y, 0.1, seed)
    if validation:    # split the training set and return validation set (not test set)
        Xs, ys = test_split(Xs[0], ys[0], 0.2, seed)
    X_scaler, y_scaler = None, None
    if preparation:
        Xs, X_scaler = data_preparation(Xs)
        ys, y_scaler = data_preparation(ys)
    datasets = map(TensorDataset, map(to_device, Xs), map(to_device, ys))
    return tuple(datasets), (X_scaler, y_scaler)


def get_datasets():
    df, X, y = get_superconductivity()
    X, y = numpy2torch(X), numpy2torch(y)
    percentiles = df[SPLIT_COLUMN].quantile([0.25, 0.75])
    domain_source = domain_split(df, X, y, -np.inf, percentiles[0.25])
    domain_target = domain_split(df, X, y, percentiles[0.75], np.inf)
    Xs_source, ys_source = test_split(*domain_source, 0.2, seed=60)
    Xs_target, ys_target = test_split(*domain_target, 0.2, seed=96)
    Xs, _ = data_preparation(Xs_source + Xs_target)    # "+" chains tuples
    ys, y_scaler = data_preparation(ys_source + ys_target)
    datasets = map(TensorDataset, map(to_device, Xs), map(to_device, ys))
    return tuple(datasets), (X_scaler, y_scaler)


def get_y(dataset):
    return torch.concat([batch[-1] for batch in DataLoader(dataset, batch_size=BS)]).cpu()


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
        for epoch in range(hyperparams["epochs"]):
            self.train_epoch(trainloader, optimiser)
            scheduler.step()
            log(self, trainset, testset, y_scaler)
        return self

    def load(self, modelname):
        self.load_state_dict(torch.load(MODELPATH.format(modelname), map_location=DEVICE))

    def save(self, modelname):
        torch.save(self.state_dict(), MODELPATH.format(modelname))


def nll(mean, variance, y):
    return F.gaussian_nll_loss(mean, y, variance, eps=EPS, full=True, reduction="none")


def clamp(tensor):
    # clamp for stability
    # https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gaussian_nll_loss
    tensor = tensor.clone()  # not to modify the original tensor
    with torch.no_grad():
        tensor.clamp_(min=EPS)
    return tensor


def standard_normal_pdf(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * x ** 2)


def standard_normal_cdf(x):
    return 0.5 + 0.5 * torch.erf(x / math.sqrt(2.0))


def normal_cdf(x, mean, variance):
    return 0.5 + 0.5 * torch.erf((x - mean) / (torch.sqrt(variance) * math.sqrt(2.0)))


def normal_pdf(x, mean, variance):
    return (1.0 / (torch.sqrt(2.0 * math.pi * variance))) * torch.exp(-0.5 * ((x - mean) ** 2 / variance))


def normal_quantile(x, mean, variance):
    return mean + torch.sqrt(variance) * math.sqrt(2) * torch.erfinv(2 * x - 1)


def normal_reliability(mean, variance, y):
    lower = [normal_quantile((1 - p) / 2, mean, variance) for p in CONFIDENCES]
    upper = [normal_quantile(1 - ((1 - p) / 2), mean, variance) for p in CONFIDENCES]
    return [torch.sum((l < y) & (y < u)) / len(y) for l, u in zip(lower, upper)]


def calibration_curve(ax, reliabilities):
    ax.plot(CONFIDENCES, reliabilities, marker="o")
    ax.plot(CONFIDENCES, CONFIDENCES)


def crps(mean, variance, y):
    variance = clamp(variance)
    std = torch.sqrt(variance)
    y = (y - mean) / std
    return std * (y * (2.0 * standard_normal_cdf(y) - 1.0) + 2.0 * standard_normal_pdf(y) - 1.0 / math.sqrt(math.pi))


def se(mean, variance, y):
    return (y - mean) ** 2


def normal_pit(mean, variance, y):
    variance = variance.clamp(min=EPS)
    return normal_cdf(y, mean, variance)


class NeuralNetwork(Model):
    def __init__(self, features, hyperparams):
        super().__init__()
        self.loss_function = {"crps": crps, "nll": nll, "se": se}[hyperparams["loss"]]
        self.model = nn.Sequential(
            nn.Linear(features, hyperparams["neurons"]),
            nn.ReLU(),
            nn.Linear(hyperparams["neurons"], 2))
        self.to(DEVICE)

    def forward(self, X):
        # torch.exp versus F.softplus output function for variance
        # mdn: exp give nans
        output = self.model(X)
        mean, variance = output[:, :1], F.softplus(output[:, 1:])
        return mean, variance

    @torch.no_grad()
    def predict(self, dataset):
        self.eval()
        output = [self(batch[0]) for batch in DataLoader(dataset, batch_size=BS)]
        mean, variance = tuple(zip(*output))
        return torch.concat(mean).cpu(), torch.concat(variance).cpu()

    def metrics(self, y_pred, y, y_scaler):
        mean, variance = y_pred
        if y_scaler is not None:
            mean = y_scaler.inverse_transform(mean)
            variance = y_scaler.inverse_transform_variance(variance)
            y = y_scaler.inverse_transform(y)
        return {
                "crps": torch.mean(crps(mean, variance, y)).item(),
                "mae": torch.mean(torch.abs(mean - y)).item(),
                "nll": torch.mean(nll(mean, variance, y)).item(),
                "pit": wandb.Histogram(normal_pit(mean, variance, y), num_bins=BINS),
                "rmse": torch.sqrt(torch.mean(torch.square(mean - y))).item(),
                "sharpness": torch.mean(variance).item(),
                "variance": wandb.Histogram(variance)}


# TODO rename
def A(mean, variance):
    std = torch.sqrt(variance)
    return 2.0 * std * standard_normal_pdf(mean / std) + mean * (2.0 * standard_normal_cdf(mean / std) - 1)


def gmm_crps(coeffs, means, variances, y):
    variances = clamp(variances)
    return torch.unsqueeze(torch.sum(coeffs * A(y - means, variances), dim=1)
            - 0.5 * torch.sum(
                (coeffs[:, :, None] * coeffs[:, None, :])
                * A(means[:, :, None] - means[:, None, :], variances[:, :, None] + variances[:, None, :]),
                dim=(1, 2)), dim=-1)


def gmm_nll(coeffs, means, variances, y):
    variances = clamp(variances)
    return -torch.log(torch.sum(coeffs * normal_pdf(y, means, variances), dim=-1, keepdim=True))
    #coeffs = clamp(coeffs)
    #return -torch.logsumexp(torch.log(coeffs) - 0.5 * torch.log(2.0 * math.pi * variances) - 0.5 * ((y - means) ** 2 / variances), dim=-1, keepdim=True)

def gmm_cdf(x, coeffs, means, variances):
    return torch.sum(coeffs * normal_cdf(x, means, variances), dim=-1, keepdim=True)


def gmm_pdf(x, coeffs, means, variances):
    return torch.sum(coeffs * normal_pdf(x, means, variances), dim=-1, keepdim=True)


def gmm_pit(coeffs, means, variances, y):
    variances = variances.clamp(min=EPS)
    return torch.sum(coeffs * normal_cdf(y, means, variances), dim=-1, keepdim=True)


class MixtureDensityNetwork(Model):
    def __init__(self, features, hyperparams):
        super().__init__()
        self.loss_function = {"crps": gmm_crps, "nll": gmm_nll}[hyperparams["loss"]]
        self.K = hyperparams["k"]
        self.model = nn.Sequential(
            nn.Linear(features, hyperparams["neurons"]),
            nn.ReLU(),
            nn.Linear(hyperparams["neurons"], 3 * self.K))
        self.to(DEVICE)

    def forward(self, X):
        output = self.model(X)
        coeffs = F.softmax(output[:, :self.K], dim=-1)
        means = output[:, self.K:(2 * self.K)]
        variances = F.softplus(output[:, (2 * self.K):])
        return coeffs, means, variances

    @torch.no_grad()
    def predict(self, dataset):
        self.eval()
        output = [self(batch[0]) for batch in DataLoader(dataset, batch_size=BS)]
        coeffs, means, variances = tuple(zip(*output))
        return torch.cat(coeffs).cpu(), torch.cat(means).cpu(), torch.cat(variances).cpu()

    def metrics(self, y_pred, y, y_scaler):
        coeffs, means, variances = y_pred
        if y_scaler is not None:
            means = y_scaler.inverse_transform(means)
            variances = y_scaler.inverse_transform_variance(variances)
            y = y_scaler.inverse_transform(y)
        return {
                "crps": torch.mean(gmm_crps(coeffs, means, variances, y)).item(),
                "nll": torch.mean(gmm_nll(coeffs, means, variances, y)).item(),
                "pit": wandb.Histogram(gmm_pit(coeffs, means, variances, y), num_bins=BINS)}


class DeepEnsemble:
    def __init__(self, features, hyperparams):
        self.models = [NeuralNetwork(features, hyperparams) for _ in range(hyperparams["m"])]

    @torch.no_grad()
    def predict(self, dataset):
        outputs = [model.predict(dataset) for model in self.models]
        means, variances = tuple(zip(*outputs))
        means = torch.concat(means, dim=1)
        variances = torch.concat(variances, dim=1)
        mean = torch.mean(means, dim=1, keepdim=True)
        variance = torch.mean(variances + torch.square(means), dim=1, keepdim=True) - torch.square(mean)
        return mean, variance

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
            model.load_state_dict(torch.load(ENSEMBLEPATH.format(modelname, i), map_location=DEVICE))

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
@click.option("--preparation", is_flag=True)
@click.option("--seed", default=50, help="Seed for train and test set split.")
@click.option("--step", default=1, help="Period of learning rate decay.")
@click.option("--validation", is_flag=True)
@click.option("--watch", is_flag=True)
@click.option("--wd", default=0.0, help="Weight decay.")
def train(**hyperparams):
    # reproducibility
    torch.manual_seed(53)
    torch.backends.cudnn.benchmark = False
    # TODO domain adaption setting
    # datasets, y_scaler = get_datasets()
    # trainset_source, testset_source, _, _ = datasets
    (trainset, testset), (_, y_scaler) = get_dataset(
            hyperparams["dataname"], hyperparams["seed"],
            hyperparams["validation"], hyperparams["preparation"])
    # generate random run name with loss information
    runname = "{}-{}-{}-{}".format(
            hyperparams["loss"], hyperparams["m"], hyperparams["k"], randint(1000, 9999))
    with wandb.init(config=hyperparams, name=runname):
        config = wandb.config
        features = trainset.tensors[0].shape[1]
        if config["m"] > 1:
            model = DeepEnsemble(features, config)
        elif config["k"] > 1:
            model = MixtureDensityNetwork(features, config)
        else:
            model = NeuralNetwork(features, config)
        if config["modelname"] is not None:
            model.load(config["modelname"])
        if config["watch"]:
            wandb.watch(model)
        model.train_epochs(trainset, testset, config, y_scaler)
        model.save(runname)


if __name__ == "__main__":
    train()
