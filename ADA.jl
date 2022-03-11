module ADA

using BSON
using CategoricalArrays
import CSV
import DataFrames
using Flux, Flux.Data, Flux.Losses, Flux.Optimise
using HDF5
using ImageCore
using Logging
import MLDatasets
using Printf
using Random
using TensorBoardLogger

export get_dr12q, get_mnist, get_svhn, prepare_mnist, prepare_svhn
export train_valid_split

export digits_al, simulate_al
export entropy_sampling, random_sampling, mcdropout_sampling
export human_labeller, oracle
export entropy, mc_mutual_information
export log2entropy

export DeepEnsembleLeNet, LeNet, MCDropoutLeNet
export SZNet
export forward, predict, probability

export accuracy, cfr
export earlystopping!, finetune!, train!
export traindigits!, trainredshift!

const WIDTH = 0.01f0
const LABELS = 0.0f0:WIDTH:6.44f0
const N_LABELS = length(LABELS)
const EDGES = -0.005f0:WIDTH:6.445f0
const STR_LABELS = [@sprintf "%.2f" label for label in LABELS]
const C = 299792.458    # the speed of light in vacuum (km / s)

include("datasets.jl")
include("active.jl")
include("models.jl")
include("train.jl")

end
