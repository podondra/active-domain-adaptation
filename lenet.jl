using Logging
using TensorBoardLogger

include("ConvNets.jl")
include("DataSets.jl")
using .ConvNets
using .DataSets

MODEL_NAME = "lenet"

X_mnist_train, y_mnist_train, X_mnist_test, y_mnist_test = prepare_mnist(get_mnist("data/mnist"))
X_mnist_train, y_mnist_train, X_mnist_valid, y_mnist_valid = train_valid_split(X_mnist_train, y_mnist_train)

X_svhn_train, y_svhn_train, X_svhn_test, y_svhn_test = prepare_svhn(get_svhn("data/svhn"))
X_svhn_train, y_svhn_train, X_svhn_valid, y_svhn_valid = train_valid_split(X_svhn_train, y_svhn_train)

model = LeNetVariant()
logger = TBLogger("runs/" * MODEL_NAME, tb_overwrite)
with_logger(logger) do
    train!(model, X_svhn_train, y_svhn_train, X_mnist_valid, y_mnist_valid;
           file_model=MODEL_NAME * ".bson")
end
