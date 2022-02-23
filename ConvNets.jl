module ConvNets

export LeNetVariant
export earlystopping!, finetune!, train!
export forward, predict
export accuracy

using BSON
using Flux, Flux.Data, Flux.Losses, Flux.Optimise

function LeNetVariant()
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
    Chain(
        # TODO The input should have 2 dimensions and reshape here.
        Conv((5, 5), 1 => 20, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 20 => 50, relu),
        Dropout(0.5),
        MaxPool((2, 2)),
        flatten,
        Dense(5 * 5 * 50, 500, relu),
        Dropout(0.5),
        Dense(500, 10)
    )
end

function forward(model_gpu, X_gpu)
    softmax(reduce(hcat, [cpu(model_gpu(x)) for x in Flux.Data.DataLoader(X_gpu, batchsize=2048)]))
end

function predict(model_gpu, X_gpu)
    Flux.onecold(forward(model_gpu, X_gpu))
end

function accuracy(y, ŷ)
    sum(y .== ŷ) / size(y, 1)
end

function finetune!(model_gpu, X_train, y_train,
        batch_size=128,    # used in Hoffman et al. (2017) and Prabhu et al. (2021)
        n_epoch=60)    # used in Prabhu et al. (2021)
    X_train_gpu = gpu(X_train)
    y_train_onehot_gpu = gpu(Flux.onehotbatch(y_train, 1:10))

    loader = DataLoader((X_train_gpu, y_train_onehot_gpu), batchsize=batch_size, shuffle=true)
    optimizer = ADAM()
    θ = params(model_gpu)
    loss_function(x, y) = logitcrossentropy(model_gpu(x), y)

    for epoch in 1:n_epoch
        Flux.Optimise.train!(loss_function, θ, loader, optimizer)
    end
end

function train!(model, X_train, y_train, X_valid, y_valid;
        batch_size=128,    # used in Hoffman et al. (2017) and Prabhu et al. (2021)
        n_epoch=60,    # used in Prabhu et al. (2021)
        file_model)
    model_gpu = gpu(model)
    X_train_gpu, X_valid_gpu = gpu(X_train), gpu(X_valid)
    y_train_onehot_gpu = gpu(Flux.onehotbatch(y_train, 1:10))

    loader = DataLoader((X_train_gpu, y_train_onehot_gpu), batchsize=batch_size, shuffle=true)
    optimizer = ADAM()
    θ = params(model_gpu)
    loss_function(x, y) = logitcrossentropy(model_gpu(x), y)

    for epoch in 1:n_epoch
        Flux.Optimise.train!(loss_function, θ, loader, optimizer)
        # logging
        ŷ_train = predict(model_gpu, X_train_gpu)
        ŷ_valid = predict(model_gpu, X_valid_gpu)
        accuracy_train = accuracy(y_train, ŷ_train)
        accuracy_valid = accuracy(y_valid, ŷ_valid)
        @info "accuracy" epoch=epoch train=accuracy_train validation=accuracy_valid
    end

    bson(file_model, model=cpu(model_gpu))
end

function earlystopping!(model, X_train, y_train, X_valid, y_valid;
        batch_size=128,    # used in Hoffman et al. (2017)
        patience=8,
        file_model)
    model_gpu = gpu(model)
    X_train_gpu, X_valid_gpu = gpu(X_train), gpu(X_valid)
    y_train_onehot_gpu = gpu(Flux.onehotbatch(y_train, 1:10))

    loader = DataLoader((X_train_gpu, y_train_onehot_gpu), batchsize=batch_size, shuffle=true)
    optimizer = ADAM()
    θ = params(model_gpu)
    loss_function(x, y) = logitcrossentropy(model_gpu(x), y)

    epoch = 0
    accuracy_valid_star = typemin(Float32) 
    i = 0
    while i < patience
        Flux.Optimise.train!(loss_function, θ, loader, optimizer)
        epoch += 1

        ŷ_train = predict(model_gpu, X_train_gpu)
        ŷ_valid = predict(model_gpu, X_valid_gpu)

        accuracy_train = accuracy(y_train, ŷ_train)
        accuracy_valid = accuracy(y_valid, ŷ_valid)
        # TODO Log loss?
        
        @info "accuracy" epoch=epoch train=accuracy_train validation=accuracy_valid

        if accuracy_valid > accuracy_valid_star
            i = 0
            accuracy_valid_star = accuracy_valid
            bson(file_model, model=cpu(model_gpu))
        else
            i += 1
        end
    end
end

end
