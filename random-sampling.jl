using BSON
using CSV
using DataFrames
using Flux
using HDF5

include("ActiveLearning.jl")
include("DataSets.jl")
using .ActiveLearning
using .DataSets

N_RUN = 30
N_QUERY_RANDOM = 10

X_mnist_train, y_mnist_train, X_mnist_test, y_mnist_test = prepare_mnist(get_mnist("data/mnist"))

random_sampling_df = DataFrame()
for run in 1:N_RUN
    rounds_random, accuracies_random = simulate_al(
                                                   random_sampling, oracle,
                                                   BSON.load("lenet.bson")[:model],
                                                   X_mnist_train, y_mnist_train,
                                                   X_mnist_test, y_mnist_test,
                                                   n_query=N_QUERY_RANDOM)
    df = DataFrame(run=run, round=rounds_random, n_query=N_QUERY_RANDOM, accuracy=accuracies_random)
    global random_sampling_df = vcat(random_sampling_df, df)
end

CSV.write("data/random-sampling.csv", random_sampling_df)
