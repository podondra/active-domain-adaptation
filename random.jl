using CSV
using DataFrames

include("ActiveDomainAdaptation.jl")
using .ActiveDomainAdaptation
include("DataSets.jl")
using .DataSets

N_RUN = 30
N_QUERY_RANDOM = 10

X_mnist_train, y_mnist_train, X_mnist_test, y_mnist_test = prepare_mnist(get_mnist("data/mnist"))

random_sampling_df = DataFrame()
for run in 1:N_RUN
    lenet = LeNetVariant("lenet.bson")
    rounds_random, accuracies_random = simulate_al(
                                                   random_sampling, oracle, lenet,
                                                   X_mnist_train, y_mnist_train,
                                                   X_mnist_test, y_mnist_test,
                                                   n_query=N_QUERY_RANDOM)
    df = DataFrame(run=run, round=rounds_random, n_query=N_QUERY_RANDOM, accuracy=accuracies_random)
    global random_sampling_df = vcat(random_sampling_df, df)
end

CSV.write("data/random.csv", random_sampling_df)
