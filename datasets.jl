function get_mnist(dir)
    MLDatasets.MNIST.traintensor(Float32, dir=dir), MLDatasets.MNIST.trainlabels(dir=dir),
    MLDatasets.MNIST.testtensor(Float32, dir=dir), MLDatasets.MNIST.testlabels(dir=dir)
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
    MLDatasets.SVHN2.traintensor(Float32, dir=dir), MLDatasets.SVHN2.trainlabels(dir=dir),
    MLDatasets.SVHN2.testtensor(Float32, dir=dir), MLDatasets.SVHN2.testlabels(dir=dir)
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

function get_dr12q(filepath)
    hdf5file = h5open(filepath)
    X_tr, z_tr = read(hdf5file, "X_tr"), read(hdf5file, "z_vi_tr")
    X_va, z_va = read(hdf5file, "X_va"), read(hdf5file, "z_vi_va")
    X_te, z_te = read(hdf5file, "X_te"), read(hdf5file, "z_vi_te")
    close(hdf5file)
    z_tr[z_tr .< 0] .= 0    # z smaller than 0 should be zero
    z_tr_categorical = cut(z_tr, EDGES, labels=STR_LABELS)
    z_tr_onehot = Flux.onehotbatch(z_tr_categorical, STR_LABELS)
    return X_tr, z_tr, z_tr_onehot, X_va, z_va, X_te, z_te
end

function get_dr16q(filepath)
    hdf5file = h5open(filepath)
    X = read(hdf5file, "X")
    idx_10k = read(hdf5file, "idx_10k")
    z_10k = read(hdf5file, "z_10k")
    close(hdf5file)
    return X[:, .~idx_10k], X[:, idx_10k], z_10k[idx_10k]
end
