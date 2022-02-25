module ActiveLearning

export simulate_al
export entropy_sampling, random_sampling
export human_labeller, oracle
export entropy

using Flux
using HDF5
using Printf
using Random

include("ConvNets.jl")
using .ConvNets

function random_sampling(index_pool, prob_pool, n_query)
    return shuffle(index_pool)[1:n_query]
end

log2entropy(x) = x != 0.0 ? log2(x) : 0.0

entropy(prob) = dropdims(-sum(prob .* log2entropy.(prob), dims=1), dims=1)

function entropy_sampling(index_pool, prob_pool, n_query)
    entr_pool = entropy(prob_pool)
    perm = sortperm(entr_pool, rev=true)
    return index_pool[perm[1:n_query]]
end

function oracle(index_query, prob_query, round)
    # Oracle labels the whole query.
    return index_query
end

function human_labeller(index_query, prob_query, round, y, file)
    entr_query = entropy(prob_query)
    h5open(file, "r+") do datafile
        write(datafile, @sprintf("index_query_%d", round), index_query)
        write(datafile, @sprintf("entr_query_%d", round), entr_query)
    end
    dataset = readline()
    datafile = h5open(file, "r")
    index_label = read(datafile, dataset)
    close(datafile)
    return index_label
end

function simulate_al(query, labeller, model, X, y, X_test, y_test;
        n_round=30, n_query=10)
    n = size(X, ndims(X))
    index_pool = Vector{Int64}(1:n)
    index_train = Vector{Int64}(undef, 0)
    
    model_gpu = gpu(model)
    X_gpu, X_test_gpu = gpu(X), gpu(X_test)
    accuracies = [accuracy(y_test, predict(model_gpu, X_test_gpu))]
    for round in 1:n_round
        # Process not only pool but all data for easier indexing.
        prob = forward(model_gpu, gpu(X_gpu))
        # Query (might be only preselection).
        index_query = query(index_pool, prob[:, index_pool], n_query)
        # Oracle (might be an inperfect human labeller that might not label the whole query).
        index_label = labeller(index_query, prob[:, index_query], round)
        # Update training set and pool.
        index_train = union(index_train, index_label)
        index_pool = setdiff(index_pool, index_label)
        # Fine-tune the active learner.
        # For other approaches see Prabhu et al. (2021) and Su et al. (2019).
        X_train, y_train = X[:, :, :, index_train], y[index_train]
        finetune!(model_gpu, X_train, y_train)
        # Evaluate performance.
        accuracy_round = accuracy(y_test, predict(model_gpu, X_test_gpu))
        @info "performance" round accuracy_round
        accuracies = vcat(accuracies, accuracy_round)
    end
    return 0:n_round, accuracies
end

end
