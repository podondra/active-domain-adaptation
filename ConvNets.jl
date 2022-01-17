module ConvNets

export ModifiedLeNet
export forward_pass, predict, train!
export accuracy

using BSON
using Flux, Flux.Data, Flux.Losses, Flux.Optimise

function ModifiedLeNet()
    Chain(
        Conv((5, 5), 1 => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 16 => 120, relu),
        flatten,
        Dense(120, 84, relu),
        # TODO Add dropout?
        Dense(84, 10)
    )
end

function forward_pass(model_gpu, X_gpu)
    softmax(reduce(hcat, [cpu(model_gpu(x)) for x in Flux.Data.DataLoader(X_gpu, batchsize=2048)]))
end

function predict(model_gpu, X_gpu)
    Flux.onecold(forward_pass(model_gpu, X_gpu))
end

function accuracy(y, ŷ)
    sum(y .== ŷ) / size(y, 1)
end

function train!(model, X_train, y_train, X_valid, y_valid;
        batch_size=64, file_model, patience=32)
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
        
        @info "accuracy" train=accuracy_train validation=accuracy_valid epoch=epoch

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
