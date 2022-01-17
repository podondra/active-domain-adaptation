module DataSets

export get_mnist, get_svhn, prepare_mnist, prepare_svhn

using ImageCore
using MLDatasets

function get_mnist(dir)
    MNIST.traintensor(Float32, dir=dir), MNIST.trainlabels(dir=dir),
    MNIST.testtensor(Float32, dir=dir), MNIST.testlabels(dir=dir)
end

function prepare_mnist(dataset_mnist)
    X_train, y_train, X_test, y_test = dataset_mnist
    X_train_new = zeros(Float32, (32, 32, 1, 60000))
    X_test_new = zeros(Float32, (32, 32, 1, 10000))
    X_train_new[3:end - 2, 3:end - 2, 1, :] = reshape(X_train, (28, 28, 1, :))
    X_test_new[3:end - 2, 3:end - 2, 1, :] = reshape(X_test, (28, 28, 1, :))
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

end
