include("ADA.jl")
using .ADA

digits_al(random_sampling, LeNet, ("models/lenet.bson", ), "data/random.csv")
