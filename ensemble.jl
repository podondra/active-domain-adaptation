include("ADA.jl")
using .ADA

filepaths = "models/" .*  ["lenet", "lenet2", "lenet3", "lenet4", "lenet5"] .* ".bson"
digits_al(entropy_sampling, DeepEnsembleLeNet, (filepaths, ), "data/ensemble.csv")
