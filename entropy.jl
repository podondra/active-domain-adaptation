include("ADA.jl")
using .ADA

digits_al(entropy_sampling, LeNet, ("models/lenet.bson", ), "data/entropy.csv")
