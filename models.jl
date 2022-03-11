function forward(model, X; batchsize=2048)
    n = size(X, ndims(X))
    batchsize = n < batchsize ? n : batchsize
    reduce(hcat, [cpu(model(x)) for x in Flux.Data.DataLoader(gpu(X); batchsize)])
end

abstract type Model end

probability(model::Model, X) = softmax(forward(model.model, X))

struct LeNet <: Model
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
LeNet() = LeNet(gpu(Chain(
                          Conv((5, 5), 1 => 20, relu),
                          MaxPool((2, 2)),
                          Conv((5, 5), 20 => 50, relu),
                          Dropout(0.5),
                          MaxPool((2, 2)),
                          flatten,
                          Dense(5 * 5 * 50, 500, relu),
                          Dropout(0.5),
                          Dense(500, 10))))

LeNet(filepath::String) = LeNet(gpu(BSON.load(filepath, @__MODULE__)[:model]))

predict(model::LeNet, X) = Flux.onecold(probability(model, X))

struct SZNet <: Model
    model
end

SZNet() = SZNet(gpu(Chain(Flux.unsqueeze(2),
                          Conv((3,), 1=>8, relu, pad=SamePad()),
                          MaxPool((2,)),
                          Conv((3,), 8=>16, relu, pad=SamePad()),
                          MaxPool((2,)),
                          Conv((3,), 16=>32, relu, pad=SamePad()),
                          Conv((3,), 32=>32, relu, pad=SamePad()),
                          MaxPool((2,)),
                          Conv((3,), 32=>64, relu, pad=SamePad()),
                          Conv((3,), 64=>64, relu, pad=SamePad()),
                          MaxPool((2,)),
                          Conv((3,), 64=>64, relu, pad=SamePad()),
                          Conv((3,), 64=>64, relu, pad=SamePad()),
                          MaxPool((2,)),
                          flatten,
                          Dense(7488, 512, relu),
                          Dropout(0.5),
                          Dense(512, 512, relu),
                          Dropout(0.5),
                          Dense(512, N_LABELS))))

SZNet(filepath::String) = SZNet(gpu(BSON.load(filepath, @__MODULE__)[:model]))

predict(model::SZNet, X) = Flux.onecold(probability(model, X), LABELS)

abstract type MCDropoutModel end

function probability(model::MCDropoutModel, X)
    trainmode!(model.model)
    return reduce(vcat, map(addfirstdim, [softmax(forward(model.model, X)) for t in 1:model.T]))
end

predict(model::MCDropoutModel, X) = Flux.onecold(dropdims(sum(probability(model, X), dims=1), dims=1))

struct MCDropoutLeNet <: MCDropoutModel
    model
    T::Int
end

MCDropoutLeNet(T::Int) = MCDropoutLeNet(gpu(Chain(
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

function MCDropoutLeNet(filepath::String, T::Int)
    MCDropoutLeNet(gpu(BSON.load(filepath, @__MODULE__)[:model]), 20)
end

abstract type DeepEnsemble end

addfirstdim = x -> reshape(x, 1, size(x)...)

function probability(ensemble::DeepEnsemble, X)
    logits = [forward(model.model, X) for model in ensemble.models]
    return sum([softmax(logit) for logit in logits]) ./ ensemble.M
end

predict(ensemble::DeepEnsemble, X) = Flux.onecold(probability(ensemble, X))

struct DeepEnsembleLeNet <: DeepEnsemble
    models
    M::Int
end

function DeepEnsembleLeNet(filepaths::Vector{String})
    DeepEnsembleLeNet([LeNet(filepath) for filepath in filepaths], length(filepaths))
end
