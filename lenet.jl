include("ConvNets.jl")
include("DataSets.jl")
using .ConvNets
using .DataSets

X_train, y_train, X_test, y_test = prepare_svhn(get_svhn("data/svhn"))
X_train, y_train, X_valid, y_valid = train_valid_split(X_train, y_train)

lenet = ModifiedLeNet()
train!(lenet, X_train, y_train, X_valid, y_valid; file_model="lenet.bson")
