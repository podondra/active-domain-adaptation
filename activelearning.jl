using Random

log2entropy(x) = x != 0.0 ? log2(x) : 0.0

entropy(prob) = dropdims(-sum(prob .* log2entropy.(prob), dims=1), dims=1)

function random_sampling(model, X_pool, index_pool, n_query)
    return shuffle(index_pool)[1:n_query]
end

function entropy_sampling(model, X_pool, index_pool, n_query)
    prob_pool = probability(model, X_pool)
    entr_pool = entropy(prob_pool)
    perm = sortperm(entr_pool, rev=true)
    return index_pool[perm[1:n_query]]
end

function mc_mutual_information(prob, T)
    prob_t = sum(prob, dims=1) / T
    entr = -sum(prob_t .* log2entropy.(prob_t), dims=2)
    exp_entr = sum(prob .* log2entropy.(prob), dims=(1, 2)) / T
    return dropdims(entr + exp_entr, dims=(1, 2))
end

function mcdropout_sampling(model, X_pool, index_pool, n_query)
    prob = probability(model, X_pool)
    mutual_information = mc_mutual_information(prob, model.T)
    perm = sortperm(mutual_information, rev=true)
    return index_pool[perm[1:n_query]]
end

function oracle(model, X_query, index_query, round)
    # Oracle labels the whole query.
    return index_query
end

function simulate_al(query, labeller, model, X, y, X_test, y_test; n_round=30, n_query=10)
    n = size(X, ndims(X))
    index_pool = Vector{Int64}(1:n)
    index_train = Vector{Int64}(undef, 0)

    accuracies = [accuracy(y_test, predict(model, X_test))]
    for round in 1:n_round
        # Query (might be only preselection).
        index_query = query(model, X[:, :, :, index_pool], index_pool, n_query)
        # Oracle (might be an inperfect human labeller that might not label the whole query).
        index_label = labeller(model, X[:, :, :, index_query], index_query, round)
        # Update training set and pool.
        index_train = union(index_train, index_label)
        index_pool = setdiff(index_pool, index_label)
        # Fine-tune the active learner.
        # For other approaches see Prabhu et al. (2021) and Su et al. (2019).
        finetune!(model, X[:, :, :, index_train], y[index_train])
        # Evaluate performance.
        accuracy_round = accuracy(y_test, predict(model, X_test))
        @info "performance" round accuracy_round
        accuracies = vcat(accuracies, accuracy_round)
    end
    return 0:n_round, accuracies
end
