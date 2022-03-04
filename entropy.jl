using CSV
using DataFrames

include("ActiveDomainAdaptation.jl")
using .ActiveDomainAdaptation
include("DataSets.jl")
using .DataSets

N_RUN = 30
N_QUERY_ENTROPY = 10

X_mnist_train, y_mnist_train, X_mnist_test, y_mnist_test = prepare_mnist(get_mnist("data/mnist"))

entropy_sampling_df = DataFrame()
for run in 1:N_RUN
    lenet = LeNetVariant("lenet.bson")
    rounds_entropy, accuracies_entropy = simulate_al(
                                                   entropy_sampling, oracle, lenet,
                                                   X_mnist_train, y_mnist_train,
                                                   X_mnist_test, y_mnist_test,
                                                   n_query=N_QUERY_ENTROPY)
    df = DataFrame(run=run, round=rounds_entropy, n_query=N_QUERY_ENTROPY, accuracy=accuracies_entropy)
    global entropy_sampling_df = vcat(entropy_sampling_df, df)
end

CSV.write("data/entropy.csv", entropy_sampling_df)
