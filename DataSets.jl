module DataSets

export get_mnist, get_svhn, prepare_mnist, prepare_svhn
export train_valid_split

using ImageCore
using MLDatasets
using Random

function get_mnist(dir)
    MNIST.traintensor(Float32, dir=dir), MNIST.trainlabels(dir=dir),
    MNIST.testtensor(Float32, dir=dir), MNIST.testlabels(dir=dir)
end

function prepare_mnist(dataset_mnist)
    X_train, y_train, X_test, y_test = dataset_mnist
    # pad with 0s to get 32 times 32 size
    X_train_new = zeros(Float32, (32, 32, 1, 60000))
    X_test_new = zeros(Float32, (32, 32, 1, 10000))
    X_train_new[3:end - 2, 3:end - 2, 1, :] = reshape(X_train, (28, 28, 1, :))
    X_test_new[3:end - 2, 3:end - 2, 1, :] = reshape(X_test, (28, 28, 1, :))
    # change label 0 to 10
    y_train[y_train .== 0] .= 10
    y_test[y_test .== 0] .= 10
    return X_train_new, y_train, X_test_new, y_test
end

function get_svhn(dir)
    SVHN2.traintensor(Float32, dir=dir), SVHN2.trainlabels(dir=dir),
    SVHN2.testtensor(Float32, dir=dir), SVHN2.testlabels(dir=dir)
end

function svhn2gray(X)
    Float32.(Gray.(colorview(RGB, permutedims(X, (3, 1, 2, 4)))))
end

function prepare_svhn(dataset_svhn)
    X_train, y_train, X_test, y_test = dataset_svhn
    X_train_new = reshape(svhn2gray(X_train), (32, 32, 1, :))
    X_test_new = reshape(svhn2gray(X_test), (32, 32, 1, :))
    return X_train_new, y_train, X_test_new, y_test
end

function train_valid_split(X, y)
    n = size(y, 1)
    n_train = round(Int, n * 0.8)    # 80 %
    Random.seed!(34)    # seed from random.org
    index_random = randperm(n)
    index_train = index_random[1:n_train]
    index_valid = index_random[n_train + 1:end]
    return X[:, :, :, index_train], y[index_train], X[:, :, :, index_valid], y[index_valid]
end

end
