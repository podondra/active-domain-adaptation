using BSON
using Flux, Flux.Data, Flux.Losses, Flux.Optimise

struct LeNetVariant
    model
end

# A variant of LeNet as presented in Hoffman et al. (2017)
# because it is used by Prabhu et al. (2021) and probably by Su et al. (2020).
# The original ConvNet uses ReLU after max pooling layers
# that is very strange if ReLU and max pooling is used (see Deep Learning book p. 336).
# Original LeNet-5 uses subsampling layers that compute average of theirs four inputs,
# multiply them by a trainable coefficient, add bias,
# and pass results through a sigmoid function (p. 2284 in LeCun et al. (1998).
# However, I think it work the same both ways.
# It also processes 28 times 28 images while it should process 32 times 32
# (see Fig. 2 in LeCun et al. (1998) and p. 2284: "The input is a 32x32 pixel image.").
# Therefore, the first hidden layer is bigger.
LeNetVariant() = LeNetVariant(gpu(Chain(
                                        Conv((5, 5), 1 => 20, relu),
                                        MaxPool((2, 2)),
                                        Conv((5, 5), 20 => 50, relu),
                                        Dropout(0.5),
                                        MaxPool((2, 2)),
                                        flatten,
                                        Dense(5 * 5 * 50, 500, relu),
                                        Dropout(0.5),
                                        Dense(500, 10))))

LeNetVariant(filepath::String) = LeNetVariant(gpu(BSON.load(filepath, @__MODULE__)[:model]))

struct MCLeNetVariant
    model
    T::Int
end

MCLeNetVariant(T::Int) = MCLeNetVariant(gpu(Chain(
                                              Conv((5, 5), 1 => 20, relu),
                                              Dropout(0.5),
                                              MaxPool((2, 2)),
                                              Conv((5, 5), 20 => 50, relu),
                                              Dropout(0.5),
                                              MaxPool((2, 2)),
                                              flatten,
                                              Dense(5 * 5 * 50, 500, relu),
                                              Dropout(0.5),
                                              Dense(500, 10))),
                                        T)

MCLeNetVariant(filepath::String, T::Int) = MCLeNetVariant(
                                                          gpu(BSON.load(filepath, @__MODULE__)[:model]),
                                                          20)

struct DeepEnsemble
    models
    M::Int
end

function forward(model, X)
    n = size(X, ndims(X))
    batchsize = n < 2048 ? n : 2048
    reduce(hcat, [cpu(model(x)) for x in Flux.Data.DataLoader(gpu(X), batchsize=batchsize)])
end

function probability(model::LeNetVariant, X)
    softmax(forward(model.model, X))
end

addfirstdim = x -> reshape(x, 1, size(x)...)

function probability(model::MCLeNetVariant, X)
    trainmode!(model.model)
    return reduce(vcat, map(addfirstdim, [softmax(forward(model.model, X)) for t in 1:model.T]))
end

function predict(model::LeNetVariant, X)
    Flux.onecold(probability(model, X))
end

function predict(model::MCLeNetVariant, X)
    Flux.onecold(dropdims(sum(probability(model, X), dims=1), dims=1))
end

accuracy(y, ŷ) = sum(y .== ŷ) / size(y, 1)

function finetune!(model, X_train, y_train,
        batchsize=128,    # used in Hoffman et al. (2017) and Prabhu et al. (2021)
        n_epoch=60)    # used in Prabhu et al. (2021)
    y_train_onehot = Flux.onehotbatch(y_train, 1:10)
    n = size(X_train, ndims(X_train))
    batchsize = n < batchsize ? n : batchsize
    loader = DataLoader((gpu(X_train), gpu(y_train_onehot)), batchsize=batchsize, shuffle=true)
    optimizer = ADAM()
    θ = params(model.model)
    loss(x, y) = logitcrossentropy(model.model(x), y)

    for epoch in 1:n_epoch
        Flux.Optimise.train!(loss, θ, loader, optimizer)
    end
end

function train!(model, X_train, y_train, X_valid, y_valid;
        batchsize=128,    # used in Hoffman et al. (2017) and Prabhu et al. (2021)
        n_epoch=60,    # used in Prabhu et al. (2021)
        file_model)
    y_train_onehot = Flux.onehotbatch(y_train, 1:10)
    loader = DataLoader((gpu(X_train), gpu(y_train_onehot)), batchsize=batchsize, shuffle=true)
    optimizer = ADAM()
    θ = params(model.model)
    loss(x, y) = logitcrossentropy(model.model(x), y)

    for epoch in 1:n_epoch
        Flux.Optimise.train!(loss, θ, loader, optimizer)
        # logging
        ŷ_train = predict(model, X_train)
        ŷ_valid = predict(model, X_valid)
        accuracy_train = accuracy(y_train, ŷ_train)
        accuracy_valid = accuracy(y_valid, ŷ_valid)
        @info "accuracy" epoch=epoch train=accuracy_train validation=accuracy_valid
    end

    bson(file_model, model=cpu(model.model))
end

function earlystopping!(model, X_train, y_train, X_valid, y_valid;
        batchsize=128,    # used in Hoffman et al. (2017)
        patience=8,
        file_model)
    y_train_onehot = Flux.onehotbatch(y_train, 1:10)
    loader = DataLoader((gpu(X_train), gpu(y_train_onehot)), batchsize=batchsize, shuffle=true)
    optimizer = ADAM()
    θ = params(model.model)
    loss(x, y) = logitcrossentropy(model.model(x), y)

    epoch = 0
    accuracy_valid_star = typemin(Float32)
    i = 0
    while i < patience
        Flux.Optimise.train!(loss, θ, loader, optimizer)
        epoch += 1

        ŷ_train = predict(model, X_train)
        ŷ_valid = predict(model, X_valid)
        accuracy_train = accuracy(y_train, ŷ_train)
        accuracy_valid = accuracy(y_valid, ŷ_valid)
        @info "accuracy" epoch=epoch train=accuracy_train validation=accuracy_valid

        if accuracy_valid > accuracy_valid_star
            i = 0
            accuracy_valid_star = accuracy_valid
            bson(file_model, model=cpu(model.model))
        else
            i += 1
        end
    end
end
