using HDF5
using Printf
using Random

log2entropy(x) = x != 0.0 ? log2(x) : 0.0

entropy(prob) = dropdims(-sum(prob .* log2entropy.(prob), dims=1), dims=1)

function random_sampling(index_pool, prob_pool, n_query)
    return shuffle(index_pool)[1:n_query]
end

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
        write(datafile, @sprintf("prob_query_%d", round), prob_query)
        write(datafile, @sprintf("entr_query_%d", round), entr_query)
    end
    dataset = readline()
    datafile = h5open(file, "r")
    index_label = read(datafile, dataset)
    close(datafile)
    return index_label
end

function simulate_al(
        query, labeller, model,
        X, y, X_test, y_test;
        n_round=30, n_query=10)
    n = size(X, ndims(X))
    index_pool = Vector{Int64}(1:n)
    index_train = Vector{Int64}(undef, 0)

    accuracies = [accuracy(y_test, predict(model, X_test))]
    for round in 1:n_round
        # Process not only pool but all data for easier indexing.
        prob = probability(model, X)
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
        finetune!(model, X_train, y_train)
        # Evaluate performance.
        accuracy_round = accuracy(y_test, predict(model, X_test))
        @info "performance" round accuracy_round
        accuracies = vcat(accuracies, accuracy_round)
    end
    return 0:n_round, accuracies
end
