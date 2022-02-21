module ActiveLearning

export simulate_al
export entropy_sampling, random_sampling
export oracle

using Flux
using Random

include("ConvNets.jl")
using .ConvNets

function random_sampling(model_gpu, X, n_query)
    n = size(X, ndims(X))
    perm = randperm(n)
    perm[1:n_query], perm[n_query + 1:end]
end

log2entropy(x) = x != 0.0 ? log2(x) : 0.0

entropy(ŷ_prob) = dropdims(-sum(ŷ_prob .* log2entropy.(ŷ_prob), dims=1), dims=1)

function entropy_sampling(model_gpu, X, n_query)
    ŷ_prob = forward(model_gpu, gpu(X))
    ŷ_entr = entropy(ŷ_prob)
    perm = sortperm(ŷ_entr, rev=true)
    perm[1:n_query], perm[n_query + 1:end]
end

function oracle(X_pool, index_query, index_pool)
    # Oracle labels the whole query.
    index_query, index_pool
end

function simulate_al(query, labeller, model, X_pool, y_pool, X_test, y_test;
        n_round=30, n_query=10)
    X_train, y_train = Array{Float32, 4}(undef, (32, 32, 1, 0)), Vector{Int64}(undef, 0)
    
    model_gpu = gpu(model)
    X_test_gpu = gpu(X_test)
    accuracies = [accuracy(y_test, predict(model_gpu, X_test_gpu))]
    for round in 1:n_round
        # Query (might be only preselection).
        index_query, index_pool = query(model_gpu, X_pool, n_query)
        # Oracle (might be an inperfect human labeller/annotator
        # that might not label/annotate the whole query).
        index_query, index_pool = labeller(X_pool, index_query, index_pool)
        X_query, y_query = X_pool[:, :, :, index_query], y_pool[index_query]
        # Update target training set and unlabelled pool.
        X_train, y_train = cat(X_train, X_query, dims=4), vcat(y_train, y_query)
        X_pool, y_pool = X_pool[:, :, :, index_pool], y_pool[index_pool]
        # Fine-tune the active learner.
        # For other approaches see Prabhu et al. (2021) and Su et al. (2019).
        model_gpu = finetune!(model_gpu, X_train, y_train)
        # Evaluate performance.
        accuracies = vcat(accuracies, accuracy(y_test, predict(model_gpu, X_test_gpu)))
    end
    return 0:n_round, accuracies
end

end
