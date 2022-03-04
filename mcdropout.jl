using CSV
using DataFrames

include("ActiveDomainAdaptation.jl")
using .ActiveDomainAdaptation
include("DataSets.jl")
using .DataSets

N_RUN = 30
N_QUERY_MC = 10

X_mnist_train, y_mnist_train, X_mnist_test, y_mnist_test = prepare_mnist(get_mnist("data/mnist"))

mc_sampling_df = DataFrame()
for run in 1:N_RUN
    mclenet = MCLeNetVariant("mclenet.bson", 20)
    rounds_mc, accuracies_mc = simulate_al(
                                           mcdropout_sampling, oracle, mclenet,
                                           X_mnist_train, y_mnist_train,
                                           X_mnist_test, y_mnist_test,
                                           n_query=N_QUERY_MC)
    df = DataFrame(run=run, round=rounds_mc, n_query=N_QUERY_MC, accuracy=accuracies_mc)
    global mc_sampling_df = vcat(mc_sampling_df, df)
    GC.gc()
end

CSV.write("data/mcdropout.csv", mc_sampling_df)
