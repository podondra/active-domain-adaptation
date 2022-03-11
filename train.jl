accuracy(y, ŷ) = sum(y .== ŷ) / length(y)

computeΔv(z, ẑ) = C .* (ẑ - z) ./ (1 .+ z)

cfr(z, ẑ; threshold=3000) = sum(abs.(computeΔv(z, ẑ)) .>= threshold) / length(z)

function finetune!(
        model, X, y_onehot;
        batchsize=128,    # used in Hoffman et al. (2017) and Prabhu et al. (2021)
        n_epoch=60)    # used in Prabhu et al. (2021)
    n = size(X, ndims(X))
    batchsize = n < batchsize ? n : batchsize
    loader = DataLoader((gpu(X), gpu(y_onehot)); batchsize, shuffle=true)
    optimizer = ADAM()
    θ = params(model.model)
    loss(x, y) = logitcrossentropy(model.model(x), y)
    for epoch in 1:n_epoch
        Flux.Optimise.train!(loss, θ, loader, optimizer)
    end
end

function finetune!(ensemble::DeepEnsemble, X, y_onehot; batchsize=128, n_epoch=60)
    for model in ensemble.models
        finetune!(model, X, y_onehot; batchsize, n_epoch)
    end
end

function train!(
        lenet, X_tr, y_tr, y_tr_onehot, X_va, y_va;
        batchsize=128,    # used in Hoffman et al. (2017) and Prabhu et al. (2021)
        n_epoch=60,    # used in Prabhu et al. (2021)
        file_model)
    loader = DataLoader((gpu(X_tr), gpu(y_tr_onehot)); batchsize, shuffle=true)
    # TODO Implement weight decay for MC dropout: Optimiser(WeightDecay(weightdecay), ADAM()).
    optimizer = ADAM()
    θ = params(lenet.model)
    loss(x, y) = logitcrossentropy(lenet.model(x), y)
    for epoch in 1:n_epoch
        Flux.Optimise.train!(loss, θ, loader, optimizer)
        accuracy_tr = accuracy(y_tr, predict(lenet, X_tr))
        accuracy_va = accuracy(y_va, predict(lenet, X_va))
        @info "accuracy" epoch=epoch train=accuracy_tr validation=accuracy_va
    end
    bson(file_model, model=cpu(lenet.model))
end

function earlystopping!(
        sznet, X_tr, y_tr, y_tr_onehot, X_va, y_va;
        batchsize=256,
        file_model,
        patience=32)
    loader = DataLoader((gpu(X_tr), gpu(y_tr_onehot)); batchsize, shuffle=true)
    optimizer = ADAM()
    θ = params(sznet.model)
    loss(x, y) = logitcrossentropy(sznet.model(x), y)
    epoch = 0
    cfr_va_star = typemax(Float32)
    i = 0
    while i < patience
        Flux.Optimise.train!(loss, θ, loader, optimizer)
        epoch += 1
        cfr_tr = cfr(y_tr, predict(sznet, X_tr))
        cfr_va = cfr(y_va, predict(sznet, X_va))
        @info "cfr" epoch=epoch train=cfr_tr validation=cfr_va
        if cfr_va < cfr_va_star
            i = 0
            cfr_va_star = cfr_va
            bson(file_model, model=cpu(sznet.model))
        else
            i += 1
        end
    end
end

function traindigits!(model, name)
    X_svhn_tr, y_svhn_tr, X_svhn_te, y_svhn_te = prepare_svhn(get_svhn("data/svhn"))
    X_mnist_tr, y_mnist_tr, X_mnist_te, y_mnist_te = prepare_mnist(get_mnist("data/mnist"))
    y_svhn_tr_onehot = Flux.onehotbatch(y_svhn_tr, 1:10)
    with_logger(TBLogger("runs/" * name, tb_overwrite)) do
        train!(model, X_svhn_tr, y_svhn_tr, y_svhn_tr_onehot, X_mnist_te, y_mnist_te;
               file_model= "models/" * name * ".bson")
    end
end

function trainredshift!(model, name)
    X_tr, z_tr, z_tr_onehot, X_va, z_va, X_te, z_te = get_dr12q("data/dr12q_superset.hdf5")
    with_logger(TBLogger("runs/" * name, tb_overwrite)) do
        earlystopping!(model, X_tr, z_tr, z_tr_onehot, X_va, z_va,
                       file_model = "models/" * name * ".bson")
    end
end
