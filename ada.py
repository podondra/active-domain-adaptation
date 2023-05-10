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

def get_boston(datapath="data/housing.data"):
    """
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    Title: Boston Housing Data
    Relevant Information: Concerns housing values in suburbs of Boston.
    Number of Instances: 506
    Number of Attributes:
       13 continuous attributes (including "class" attribute "MEDV"), 1 binary-valued attribute.
    Attribute Information:
       1. CRIM      per capita crime rate by town
       2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
       3. INDUS     proportion of non-retail business acres per town
       4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
       5. NOX       nitric oxides concentration (parts per 10 million)
       6. RM        average number of rooms per dwelling
       7. AGE       proportion of owner-occupied units built prior to 1940
       8. DIS       weighted distances to five Boston employment centres
       9. RAD       index of accessibility to radial highways
       10. TAX      full-value property-tax rate per $10,000
       11. PTRATIO  pupil-teacher ratio by town
       12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
       13. LSTAT    % lower status of the population
       14. MEDV     Median value of owner-occupied homes in $1000's
    """
    df = pd.read_csv(datapath, dtype=np.float32, header=None, sep="\s+")
    return df, df.values[:, :13], df.values[:, 13:]


def get_concrete(datapath="data/Concrete_Data.xls"):
    """
    https://archive.ics.uci.edu/ml/datasets/Concrete%20Compressive%20Strength
    Data Set Information:
        Number of instances 1030
        Number of Attributes 9
        Attribute breakdown 8 quantitative input variables, and 1 quantitative output variable
    Attribute Information:
        Given are the variable name, variable type, the measurement unit and a brief description.
        The concrete compressive strength is the regression problem.
        The order of this listing corresponds to the order of numerals along the rows of the database.
        Name -- Data Type -- Measurement -- Description
        Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
        Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
        Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
        Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
        Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
        Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
        Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
        Age -- quantitative -- Day (1~365) -- Input Variable
        Concrete compressive strength -- quantitative -- MPa -- Output Variable
    """
    df = pd.read_excel(datapath, dtype=np.float32)
    return df, df.values[:, :8], df.values[:, 8:]


# TODO There are two output variables.
def get_energy(datapath="data/ENB2012_data.xlsx"):
    """
    https://archive.ics.uci.edu/ml/datasets/energy+efficiency
    According to https://github.com/yaringal/DropoutUncertaintyExps, the first one it the one.
    https://archive.ics.uci.edu/ml/datasets/energy+efficiency
    Data Set Information:
      We perform energy analysis using 12 different building shapes simulated in Ecotect.
      The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters.
      We simulate various settings as functions of the afore-mentioned characteristics to obtain 768 building shapes.
      The dataset comprises 768 samples and 8 features, aiming to predict two real valued responses.
      It can also be used as a multi-class classification problem if the response is rounded to the nearest integer.
    Attribute Information:
      The dataset contains eight attributes (or features, denoted by X1...X8) and two responses (or outcomes, denoted by y1 and y2).
      The aim is to use the eight features to predict each of the two responses.
      Specifically:
      X1 Relative Compactness
      X2 Surface Area
      X3 Wall Area
      X4 Roof Area
      X5 Overall Height
      X6 Orientation
      X7 Glazing Area
      X8 Glazing Area Distribution
      y1 Heating Load
      y2 Cooling Load
    """
    df = pd.read_excel(datapath, dtype=np.float32)
    return df, df.values[:, :8], df.values[:, 8:9]


def get_kin8nm(datapath="data/Dataset.data"):
    df = pd.read_csv(datapath, dtype=np.float32, header=None, sep="\s+")
    return df, df.values[:, :8], df.values[:, 8:]


# TODO there are again two target variables
def get_naval(datapath="data/data.txt"):
    """
    https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
    Data Set Information:
      The experiments have been carried out by means of a numerical simulator of a naval vessel (Frigate) characterized by a Gas Turbine (GT) propulsion plant.
      The different blocks forming the complete simulator (Propeller, Hull, GT, Gear Box and Controller) have been developed and fine tuned over the year on several similar real propulsion plants.
      In view of these observations the available data are in agreement with a possible real vessel.
      In this release of the simulator it is also possible to take into account the performance decay over time of the GT components such as GT compressor and turbines.
      The propulsion system behaviour has been described with this parameters:
      - Ship speed (linear function of the lever position lp).
      - Compressor degradation coefficient kMc.
      - Turbine degradation coefficient kMt.
      so that each possible degradation state can be described by a combination of this triple (lp,kMt,kMc).
      The range of decay of compressor and turbine has been sampled with an uniform grid of precision 0.001 so to have a good granularity of representation.
      In particular for the compressor decay state discretization the kMc coefficient has been investigated in the domain [1; 0.95], and the turbine coefficient in the domain [1; 0.975].
      Ship speed has been investigated sampling the range of feasible speed from 3 knots to 27 knots with a granularity of representation equal to tree knots.
      A series of measures (16 features) which indirectly represents of the state of the system subject to performance decay has been acquired and stored in the dataset over the parameter's space.
    Attribute Information:
      - A 16-feature vector containing the GT measures at steady state of the physical asset:
      Lever position (lp) [ ]
      Ship speed (v) [knots]
      Gas Turbine (GT) shaft torque (GTT) [kN m]
      GT rate of revolutions (GTn) [rpm]
      Gas Generator rate of revolutions (GGn) [rpm]
      Starboard Propeller Torque (Ts) [kN]
      Port Propeller Torque (Tp) [kN]
      Hight Pressure (HP) Turbine exit temperature (T48) [C]
      GT Compressor inlet air temperature (T1) [C]
      GT Compressor outlet air temperature (T2) [C]
      HP Turbine exit pressure (P48) [bar]
      GT Compressor inlet air pressure (P1) [bar]
      GT Compressor outlet air pressure (P2) [bar]
      GT exhaust gas pressure (Pexh) [bar]
      Turbine Injecton Control (TIC) [%]
      Fuel flow (mf) [kg/s]
      - GT Compressor decay state coefficient
      - GT Turbine decay state coefficient
    """
    df = pd.read_csv(datapath, dtype=np.float32, header=None, sep="\s+")
    return df, df.values[:, :16], df.values[:, 16:17]


def get_power(datapath="data/Folds5x2_pp.xlsx"):
    # https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
    df = pd.read_excel(datapath, dtype=np.float32)
    return df, df.values[:, :4], df.values[:, 4:]


def get_protein(datapath="data/CASP.csv"):
    """https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
    RMSD-Size of the residue.
    F1 - Total surface area.
    F2 - Non polar exposed area.
    F3 - Fractional area of exposed non polar residue.
    F4 - Fractional area of exposed non polar part of residue.
    F5 - Molecular mass weighted exposed area.
    F6 - Average deviation from standard exposed area of residue.
    F7 - Euclidian distance.
    F8 - Secondary structure penalty.
    F9 - Spacial Distribution constraints (N,K Value)."""
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
    """
    https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    Data Set Information:
        The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
        For more details, consult Cortez et al. [2009].
        Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output)
        variables are available
        (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

        These datasets can be viewed as classification or regression tasks.
        The classes are ordered and not balanced
        (e.g. there are many more normal wines than excellent or poor ones).
        Outlier detection algorithms could be used to detect the few excellent or poor wines.
        Also, we are not sure if all input variables are relevant.
        So it could be interesting to test feature selection methods.

    Attribute Information:
        Input variables (based on physicochemical tests):
        1 - fixed acidity
        2 - volatile acidity
        3 - citric acid
        4 - residual sugar
        5 - chlorides
        6 - free sulfur dioxide
        7 - total sulfur dioxide
        8 - density
        9 - pH
        10 - sulphates
        11 - alcohol
        Output variable (based on sensory data):
        12 - quality (score between 0 and 10)
    """
    df = pd.read_csv(datapath, dtype=np.float32, sep=";")
    return df, df.values[:, :11], df.values[:, 11:]


def get_yacht(datapath="data/yacht_hydrodynamics.data"):
    """
    https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics
    Delft data set, used to predict the hydodynamic performance of sailing yachts
    from dimensions and velocity.

    Data Set Information:

    Prediction of residuary resistance of sailing yachts at the initial design stage
    is of a great value for evaluating the ship's performance
    and for estimating the required propulsive power.
    Essential inputs include the basic hull dimensions and the boat velocity.

    The Delft data set comprises 308 full-scale experiments,
    which were performed at the Delft Ship Hydromechanics Laboratory for that purpose.
    These experiments include 22 different hull forms,
    derived from a parent form closely related to the 'Standfast' designed by Frans Maas.

    Attribute Information:

    Variations concern hull geometry coefficients and the Froude number:

    1 gey. Longitudinal position of the center of buoyancy, adimensional.
    2. Prismatic coefficient, adimensional.
    3. Length-displacement ratio, adimensional.
    4. Beam-draught ratio, adimensional.
    5. Length-beam ratio, adimensional.
    6. Froude number, adimensional.

    The measured variable is the residuary resistance per unit weight of displacement:

    7. Residuary resistance per unit weight of displacement, adimensional.
    """
    df = pd.read_csv(datapath, dtype=np.float32, header=None, sep="\s+")
    return df, df.values[:, :6], df.values[:, 6:]


# TODO You should respect the following train / test split:
# train: first 463,715 examples
# test: last 51,630 examples
# It avoids the 'producer effect' by making sure no song
# from a given artist ends up in both the train and test set.
def get_year(datapath="data/YearPredictionMSD.txt"):
    """
    https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    Attribute Information:
        90 attributes, 12 = timbre average, 78 = timbre covariance
        The first value is the year (target), ranging from 1922 to 2011.
        Features extracted from the 'timbre' features from The Echo Nest API.
        We take the average and covariance over all 'segments', each segment
        being described by a 12-dimensional timbre vector.
    """
    df = pd.read_csv(datapath, dtype=np.float32, header=None)
    return df, df.values[:, 1:], df.values[:, :1]


DATASETS = {
        "boston": get_boston,
        "concrete": get_concrete,
        "energy": get_energy,
        "inverse": get_inverse,
        "kin8nm": get_kin8nm,
        "naval": get_naval,
        "power": get_power,
        "protein": get_protein,
        "superconductivity": get_superconductivity,
        "wine": get_wine,
        "yacht": get_yacht,
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
        self.std[self.std == 0.0] = 1.0
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
        Xs, ys = test_split(Xs[0], ys[0], 0.1, seed)
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
    return 0.5 + 0.5 * torch.erf((x - mean) / (torch.sqrt(clamp(variance)) * math.sqrt(2.0)))


def normal_pdf(x, mean, variance):
    variance = clamp(variance)
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
    y = (y - mean) / torch.sqrt(clamp(variance))
    return torch.sqrt(variance) * (y * (2.0 * standard_normal_cdf(y) - 1.0) + 2.0 * standard_normal_pdf(y) - 1.0 / math.sqrt(math.pi))


def se(mean, variance, y):
    return (y - mean) ** 2


def normal_pit(mean, variance, y):
    return normal_cdf(y, mean, variance)


class NeuralNetwork(Model):
    def __init__(self, features, hyperparams):
        super().__init__()
        self.loss_function = {"crps": crps, "nll": nll, "se": se}[hyperparams["loss"]]
        layers = [nn.Linear(features, hyperparams["neurons"]), nn.ReLU()]
        for _ in range(hyperparams["hiddens"] - 1):
            layers += [nn.Linear(hyperparams["neurons"], hyperparams["neurons"]), nn.ReLU()]
        layers += [nn.Linear(hyperparams["neurons"], 2)]
        self.model = nn.Sequential(*layers)
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
    mean_std = mean / torch.sqrt(clamp(variance))
    return 2.0 * torch.sqrt(variance) * standard_normal_pdf(mean_std) + mean * (2.0 * standard_normal_cdf(mean_std) - 1)


def gmm_crps(coeffs, means, variances, y):
    return torch.unsqueeze(torch.sum(coeffs * A(y - means, variances), dim=1)
            - 0.5 * torch.sum(
                (coeffs[:, :, None] * coeffs[:, None, :])
                * A(means[:, :, None] - means[:, None, :], variances[:, :, None] + variances[:, None, :]),
                dim=(1, 2)), dim=-1)


def gmm_nll(coeffs, means, variances, y):
    variances = clamp(variances)
    coeffs = clamp(coeffs)
    return -torch.logsumexp(
            torch.log(coeffs)
            - 0.5 * torch.log(2.0 * math.pi * variances)
            - 0.5 * ((y - means) ** 2 / variances),
            dim=-1, keepdim=True)
    #return -torch.log(torch.sum(coeffs * normal_pdf(y, means, variances), dim=-1, keepdim=True))


def gmm_cdf(x, coeffs, means, variances):
    return torch.sum(coeffs * normal_cdf(x, means, variances), dim=-1, keepdim=True)


def gmm_pdf(x, coeffs, means, variances):
    return torch.sum(coeffs * normal_pdf(x, means, variances), dim=-1, keepdim=True)


def gmm_pit(coeffs, means, variances, y):
    return torch.sum(coeffs * normal_cdf(y, means, variances), dim=-1, keepdim=True)


class MixtureDensityNetwork(Model):
    def __init__(self, features, hyperparams):
        super().__init__()
        self.loss_function = {"crps": gmm_crps, "nll": gmm_nll}[hyperparams["loss"]]
        self.K = hyperparams["k"]
        layers = [nn.Linear(features, hyperparams["neurons"]), nn.ReLU()]
        for _ in range(hyperparams["hiddens"] - 1):
            layers += [nn.Linear(hyperparams["neurons"], hyperparams["neurons"]), nn.ReLU()]
        layers += [nn.Linear(hyperparams["neurons"], 3 * self.K)]
        self.model = nn.Sequential(*layers)
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
        self.M = hyperparams["m"]
        self.models = [NeuralNetwork(features, hyperparams) for _ in range(self.M)]

    @torch.no_grad()
    def predict(self, dataset):
        outputs = [model.predict(dataset) for model in self.models]
        means, variances = tuple(zip(*outputs))
        means = torch.concat(means, dim=1)
        variances = torch.concat(variances, dim=1)
        return means, variances
        # TODO mean = torch.mean(means, dim=1, keepdim=True)
        # TODO variance = torch.mean(variances + torch.square(means), dim=1, keepdim=True) - torch.square(mean)
        # TODO return mean, variance

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

    def metrics(self, y_pred, y, y_scaler):
        means, variances = y_pred
        if y_scaler is not None:
            means = y_scaler.inverse_transform(means)
            variances = y_scaler.inverse_transform_variance(variances)
            y = y_scaler.inverse_transform(y)
        coeffs = torch.full_like(means, 1 / self.M)
        return {
                "crps": torch.mean(gmm_crps(coeffs, means, variances, y)).item(),
                "nll": torch.mean(gmm_nll(coeffs, means, variances, y)).item(),
                "pit": wandb.Histogram(gmm_pit(coeffs, means, variances, y), num_bins=BINS)}

    def load(self, modelname):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(ENSEMBLEPATH.format(modelname, i), map_location=DEVICE))

    def save(self, modelname):
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), ENSEMBLEPATH.format(modelname, i))


@click.command()
@click.option("--bs", default=100, help="Batch size.")
@click.option("--dataname", required=True, help="Data set name.")
@click.option("--epochs", default=1000, help="Number of epochs.")
@click.option("--gamma", default=1.0, help="Multiplicative factor of learning rate decay.")
@click.option("--hiddens", default=1, help="Number of hidden layers.")
@click.option("--k", default=1, help="Number of MDN components.")
@click.option("--loss", default="nll", help="Loss function.")
@click.option("--lr", default=0.001, help="Learning rate.")
@click.option("--neurons", default=50, help="Number of neurons in the first hidden layer.")
@click.option("--m", default=1, help="Number of model in deep ensemble.")
@click.option("--modelname")
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
