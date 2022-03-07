using Logging
using TensorBoardLogger

include("ActiveDomainAdaptation.jl")
using .ActiveDomainAdaptation
include("DataSets.jl")
using .DataSets

function train_ensemble(model_name)
    X_svhn_train, y_svhn_train, X_svhn_test, y_svhn_test = prepare_svhn(get_svhn("data/svhn"))
    X_mnist_train, y_mnist_train, X_mnist_test, y_mnist_test = prepare_mnist(get_mnist("data/mnist"))

    model = LeNetVariant()
    logger = TBLogger("runs/" * model_name, tb_overwrite)
    with_logger(logger) do
        train!(
               model, X_svhn_train, y_svhn_train, X_mnist_test, y_mnist_test;
               file_model=model_name * ".bson")
    end
end
