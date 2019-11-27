import os
import pickle
import ssl
import struct
import tarfile
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import cv2
from scipy import io, misc

from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds


def get_byte(file_in):
    int_out = ord(file_in.read(1))
    return int_out


def get_int(file_in):
    int_out = struct.unpack('>i', file_in.read(4))[0]
    return int_out


def get_image(file_in, row=28, col=28):
    raw_data = file_in.read(row * col)
    out_image = np.frombuffer(raw_data, np.uint8)
    out_image = out_image.reshape((28, 28))
    return out_image


def get_caltech101(save_dir=None, root_path=None):
    assert ((save_dir is not None and root_path is None) or (save_dir is None and root_path is not None))

    if root_path is None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        print('Downloading Caltech101 dataset...')
        tar_path = os.path.join(save_dir, "101_ObjectCategories.tar.gz")
        url = urllib.request.URLopener(context=ctx)
        url.retrieve("https://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz", tar_path)
        print('Download Done, Extracting...')
        tar = tarfile.open(tar_path)
        tar.extractall(save_dir)
        tar.close()

    root = os.path.join(save_dir, "101_ObjectCategories") if not root_path else root_path

    train_x = []
    train_y = []
    val_x = []
    val_y = []

    label = 0
    for cls_folder in os.listdir(root):
        cls_root = os.path.join(root, cls_folder)
        if not os.path.isdir(cls_root):
            continue

        cls_images = [misc.imread(os.path.join(cls_root, img_name)) for img_name in os.listdir(cls_root)]
        cls_images = [np.repeat(np.expand_dims(img, 2), 3, axis=2) if len(img.shape) == 2 else img for img in
                      cls_images]
        cls_images = np.array([np.reshape(cv2.resize(img, dsize=(224, 224)), (3, 224, 224)) for img in cls_images])
        new_index = np.random.permutation(np.arange(cls_images.shape[0]))
        cls_images = cls_images[new_index, :, :, :]

        train_x.append(cls_images[:30])
        train_y.append(np.array([label] * 30))
        if len(cls_images) <= 80:
            val_x.append(cls_images[30:])
            val_y.append(np.array([label] * (len(cls_images) - 30)))
        else:
            val_x.append(cls_images[30:80])
            val_y.append(np.array([label] * 50))
        label += 1

    Xtr = np.concatenate(train_x)
    Ytr = np.concatenate(train_y)
    Xval = np.concatenate(val_x)
    Yval = np.concatenate(val_y)

    print('Xtr shape ', Xtr.shape)
    print('Ytr shape ', Ytr.shape)
    print('Xval shape ', Xval.shape)
    print('Yval shape ', Yval.shape)

    return Xtr, Ytr, Xval, Yval


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).astype(np.uint8)
        Y = np.array(Y, dtype=np.int64)
        return X, Y


def get_cifar10(save_dir=None, root_path=None):
    ''' If root_path is None, we download the data set from internet.

        Either save path or root path must not be None and not both.

        Returns Xtr, Ytr, Xte, Yte as numpy arrays
    '''

    assert ((save_dir is not None and root_path is None) or (save_dir is None and root_path is not None))

    if root_path is None:
        print('Downloading CIFAR10 dataset...')
        tar_path = os.path.join(save_dir, "cifar-10-python.tar.gz")
        url = urllib.request.URLopener()
        url.retrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tar_path)
        print('Download Done, Extracting...')
        tar = tarfile.open(tar_path)
        tar.extractall(save_dir)
        tar.close()

    root = os.path.join(save_dir, "cifar-10-batches-py") if not root_path else root_path

    # Training Data
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    print('Xtrain shape', Xtr.shape)
    print('Ytrain shape', Ytr.shape)

    # Testing data
    Xte, Yte = load_CIFAR_batch(os.path.join(root, 'test_batch'))
    print('Xtest shape', Xte.shape)
    print('Ytest shape', Yte.shape)

    return Xtr, Ytr, Xte, Yte


def load_cifar100_data(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        data = datadict[b'data']
        labels = datadict[b'fine_labels']
        N = len(labels)
        data = data.reshape((N, 3, 32, 32))
        labels = np.array(labels)

        return data, labels


def get_cifar100(save_dir=None, root_path=None):
    ''' If root_path is None, we download the data set from internet.

        Either save path or root path must not be None and not both.

        Returns Xtr, Ytr, Xte, Yte as numpy arrays
    '''

    assert ((save_dir is not None and root_path is None) or (save_dir is None and root_path is not None))

    if root_path is None:
        print('Downloading CIFAR100 dataset...')
        tar_path = os.path.join(save_dir, "cifar-100-python.tar.gz")
        url = urllib.request.URLopener()
        url.retrieve("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", tar_path)
        print('Download Done, Extracting...')
        tar = tarfile.open(tar_path)
        tar.extractall(save_dir)
        tar.close()

    root = os.path.join(save_dir, "cifar-100-python") if not root_path else root_path

    Xtr, Ytr = load_cifar100_data(os.path.join(root, 'train'))
    Xte, Yte = load_cifar100_data(os.path.join(root, 'test'))

    print('Xtrain shape', Xtr.shape)
    print('Ytrain shape', Ytr.shape)
    print('Xtest shape', Xte.shape)
    print('Ytest shape', Yte.shape)

    return Xtr, Ytr, Xte, Yte


def load_mnist(image_fname, label_fname):
    with open(image_fname, "rb") as image_file, open(label_fname, "rb") as label_file:
        assert (get_int(image_file) == 2051)
        assert (get_int(label_file) == 2049)

        n_items_label = get_int(label_file)
        n_items = get_int(image_file)
        assert (n_items_label == n_items)
        assert (get_int(image_file) == 28)
        assert (get_int(image_file) == 28)

        Y = []
        X = np.zeros((n_items, 1, 28, 28), dtype=np.uint8)
        print("Reading [%d] items" % n_items)
        for i in range(n_items):
            label = get_byte(label_file)
            assert (label <= 9)
            assert (label >= 0)
            Y.append(label)
            X[i, :] = get_image(image_file)
    return X, np.asarray(Y)


def get_mnist(save_dir=None, root_path=None):
    ''' If root_path is None, we download the data set from internet.

        Either save path or root path must not be None and not both.

        Returns Xtr, Ytr, Xte, Yte as numpy arrays
    '''

    assert ((save_dir is not None and root_path is None) or (save_dir is None and root_path is not None))

    mnist_files = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                   't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
    out_mnist_files = []
    if root_path is None:
        print('Downloading MNIST dataset...')
        for fname in mnist_files:
            out_file = os.path.join(save_dir, "%s" % fname)
            tar_path = os.path.join(save_dir, "%s.gz" % fname)
            out_mnist_files.append(out_file)
            url = urllib.request.URLopener()
            url.retrieve("http://yann.lecun.com/exdb/mnist/%s.gz" % fname, tar_path)
            print('Download Done, Extracting... [%s]' % tar_path)
            os.system('gunzip -f %s' % tar_path)

    Xtr, Ytr = load_mnist(out_mnist_files[0], out_mnist_files[1])
    print('Xtrain shape', Xtr.shape)
    print('Ytrain shape', Ytr.shape)

    # Testing data
    Xte, Yte = load_mnist(out_mnist_files[2], out_mnist_files[3])
    print('Xtest shape', Xte.shape)
    print('Ytest shape', Yte.shape)

    return Xtr, Ytr, Xte, Yte


def get_svhn(save_dir=None, root_path=None):
    ''' If root_path is None, we download the data set from internet.

        Either save path or root path must not be None and not both.

        Returns Xtr, Ytr, Xte, Yte as numpy arrays
    '''

    assert ((save_dir is not None and root_path is None) or (save_dir is None and root_path is not None))

    if root_path is None:
        new_save_dir = os.path.join(save_dir, 'og_data')
        if not os.path.isdir(new_save_dir):
            os.mkdir(new_save_dir)
        train_mat = os.path.join(new_save_dir, "train_32x32.mat")
        test_mat = os.path.join(new_save_dir, "test_32x32.mat")
        url = urllib.request.URLopener()

        print('Downloading Svhn Train...')
        url.retrieve("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", train_mat)
        print('Downloading Svhn Test...')
        url.retrieve("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", test_mat)

    root = new_save_dir if not root_path else root_path

    train = io.loadmat(os.path.join(root, 'train_32x32.mat'))
    Xtr = train['X']
    Ytr = train['y']
    del train

    test = io.loadmat(os.path.join(root, 'test_32x32.mat'))
    Xte = test['X']
    Yte = test['y']
    del test

    Xtr = np.transpose(Xtr, (3, 2, 0, 1))
    Xte = np.transpose(Xte, (3, 2, 0, 1))
    Ytr = Ytr.reshape(Ytr.shape[:1]) - 1
    Yte = Yte.reshape(Yte.shape[:1]) - 1

    print('Xtrain shape', Xtr.shape)
    print('Ytrain shape', Ytr.shape)
    print('Xtest shape', Xte.shape)
    print('Ytest shape', Yte.shape)

    return Xtr, Ytr, Xte, Yte


def get_svhn_small(save_dir=None, root_path=None):
    assert ((save_dir is not None and root_path is None) or (save_dir is None and root_path is not None))
    Xtr, Ytr, Xte, Yte = get_svhn(save_dir, root_path)

    val_x = []
    val_y = []
    train_x = []
    train_y = []
    for i in np.unique(Ytr):
        # Get 400 images from X_small
        X_small_label = Xtr[Ytr == i]
        train_x.append(X_small_label[:700])
        train_y.append([i] * 700)
        val_x.append(X_small_label[700:1000])
        val_y.append([i] * 300)

    Xtr = np.concatenate(train_x)
    Ytr = np.concatenate(train_y)
    Xval = np.concatenate(val_x)
    Yval = np.concatenate(val_y)

    print('Xtrain shape', Xtr.shape)
    print('Ytrain shape', Ytr.shape)
    print('Xval shape', Xval.shape)
    print('Yval shape', Yval.shape)
    print('Xtest shape', Xte.shape)
    print('Ytest shape', Yte.shape)

    return Xtr, Ytr, Xval, Yval, Xte, Yte


def get_svhn_full(save_dir=None, root_path=None):
    ''' If root_path is None, we download the data set from internet.

        Either save path or root path must not be None and not both.

        Returns Xtr, Ytr, Xte, Yte as numpy arrays
    '''

    assert ((save_dir is not None and root_path is None) or (save_dir is None and root_path is not None))

    Xtr_small, Ytr_small, Xte, Yte = get_svhn(save_dir, root_path)

    if root_path is None:
        new_save_dir = os.path.join(save_dir, 'og_data')
        if not os.path.isdir(new_save_dir):
            os.mkdir(new_save_dir)
        extra_mat = os.path.join(new_save_dir, "extra_32x32.mat")
        url = urllib.request.URLopener()

        print('Downloading Svhn Extra...')
        url.retrieve("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", extra_mat)

    root = new_save_dir if not root_path else root_path
    extra = io.loadmat(os.path.join(root, 'extra_32x32.mat'))
    Xtr_extra = extra['X']
    Ytr_extra = extra['y']

    Xtr_extra = np.transpose(Xtr_extra, (3, 2, 0, 1))
    Ytr_extra = Ytr_extra.reshape(Ytr_extra.shape[:1]) - 1

    print('Xextra shape', Xtr_extra.shape)
    print('Yextra shape', Ytr_extra.shape)

    val_x = []
    val_y = []
    train_x = []
    train_y = []
    for i in np.unique(Ytr_small):
        # Get 400 images from X_small
        X_small_label = Xtr_small[Ytr_small == i]
        val_x.append(X_small_label[:400])
        val_y.append([i] * 400)
        train_x.append(X_small_label[400:])
        train_y.append([i] * (X_small_label.shape[0] - 400))
        # Get 200 images from X_small
        X_extra_label = Xtr_extra[Ytr_extra == i]
        val_x.append(X_extra_label[:200])
        val_y.append([i] * 200)
        train_x.append(X_extra_label[200:])
        train_y.append([i] * (X_extra_label.shape[0] - 200))

    Xtr = np.concatenate(train_x)
    Ytr = np.concatenate(train_y)
    Xval = np.concatenate(val_x)
    Yval = np.concatenate(val_y)

    return Xtr, Ytr, Xval, Yval, Xte, Yte


def get_fashion_mnist():
    dataset_train = tfds.as_numpy(tfds.load(name="fashion_mnist", split=tfds.Split.TRAIN, batch_size=-1))
    dataset_test = tfds.as_numpy(tfds.load(name="fashion_mnist", split=tfds.Split.TEST, batch_size=-1))

    x_train, y_train = dataset_train["image"], dataset_train["label"]
    x_test, y_test = dataset_test["image"], dataset_test["label"]

    # format dataset to channels x height x width
    x_train = np.rollaxis(x_train, 3, 1)
    x_test = np.rollaxis(x_test, 3, 1)

    print('Xtrain shape', x_train.shape)
    print('Ytrain shape', y_train.shape)
    print('Xtest shape', x_test.shape)
    print('Ytest shape', y_test.shape)

    return x_train, y_train, x_test, y_test


def get_flowers_102():
    dataset_train = tfds.as_numpy(tfds.load(name="oxford_flowers102", split=tfds.Split.TRAIN, batch_size=-1))
    dataset_validation = tfds.as_numpy(tfds.load(name="oxford_flowers102", split=tfds.Split.VALIDATION, batch_size=-1))
    dataset_test = tfds.as_numpy(tfds.load(name="oxford_flowers102", split=tfds.Split.TEST, batch_size=-1))

    x_train, y_train = dataset_train["image"], dataset_train["label"]
    x_validation, y_validation = dataset_validation["image"], dataset_validation["label"]
    x_test, y_test = dataset_test["image"], dataset_test["label"]

    # resize images
    x_train = np.array([cv2.resize(img, dsize=(32, 32)) for img in x_train])
    x_validation = np.array([cv2.resize(img, dsize=(32, 32)) for img in x_validation])
    x_test = np.array([cv2.resize(img, dsize=(32, 32)) for img in x_test])

    # format dataset to channels x height x width
    x_train = np.rollaxis(x_train, 3, 1)
    x_validation = np.rollaxis(x_validation, 3, 1)
    x_test = np.rollaxis(x_test, 3, 1)

    print('Xtrain shape', x_train.shape)
    print('Ytrain shape', y_train.shape)
    print('Xval shape', x_validation.shape)
    print('Yval shape', y_validation.shape)
    print('Xtest shape', x_test.shape)
    print('Ytest shape', y_test.shape)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def get_flowers_5():
    dataset = tfds.as_numpy(tfds.load(name="tf_flowers", split=tfds.Split.TRAIN, batch_size=-1))
    x, y = dataset["image"], dataset["label"]

    # resize images
    x = np.array([cv2.resize(img, dsize=(32, 32)) for img in x])

    # format dataset to channels x height x width
    x = np.rollaxis(x, 3, 1)

    # Split Percentages: train = 80%, validation  = 10%, test is 10%
    x_train, x, y_train, y = train_test_split(x, y, test_size=0.2, shuffle=False)
    x_validation, x_test, y_validation, y_test = train_test_split(x, y, test_size=0.5, shuffle=False)

    print('Xtrain shape', x_train.shape)
    print('Ytrain shape', y_train.shape)
    print('Xval shape', x_validation.shape)
    print('Yval shape', y_validation.shape)
    print('Xtest shape', x_test.shape)
    print('Ytest shape', y_test.shape)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def get_food_101():
    dataset = tfds.as_numpy(tfds.load("food101", split=tfds.Split.TRAIN))
    x = []
    y = []
    for example in dataset:
        # resize images
        x.append(cv2.resize(example["image"], dsize=(32, 32)))
        y.append(example["label"])
    x, y = np.array(x), np.array(y)

    # format dataset to channels x height x width
    x = np.rollaxis(x, 3, 1)

    # Split Percentages: train = 80%, validation  = 10%, test is 10%
    x_train, x, y_train, y = train_test_split(x, y, test_size=0.2, shuffle=False)
    x_validation, x_test, y_validation, y_test = train_test_split(x, y, test_size=0.5, shuffle=False)

    print('Xtrain shape', x_train.shape)
    print('Ytrain shape', y_train.shape)
    print('Xval shape', x_validation.shape)
    print('Yval shape', y_validation.shape)
    print('Xtest shape', x_test.shape)
    print('Ytest shape', y_test.shape)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def load_stl_10(path):
    stl_10_images = [os.path.join(path, filename) for filename in ['train_X.bin', 'test_X.bin']]
    stl_10_labels = [os.path.join(path, filename) for filename in ['train_y.bin', 'test_y.bin']]

    with open(stl_10_images[0], 'rb') as file:
        x_train = np.transpose(np.reshape(np.fromfile(file, dtype=np.uint8), (-1, 3, 96, 96)), (0, 3, 2, 1))
    with open(stl_10_labels[0], 'rb') as file:
        y_train = np.fromfile(file, dtype=np.uint8)
    with open(stl_10_images[1], 'rb') as file:
        x_test = np.transpose(np.reshape(np.fromfile(file, dtype=np.uint8), (-1, 3, 96, 96)), (0, 3, 2, 1))
    with open(stl_10_labels[1], 'rb') as file:
        y_test = np.fromfile(file, dtype=np.uint8)

    return x_train, y_train, x_test, y_test


def get_stl_10(save_dir=None, root_path=None):
    ''' If root_path is None, we download the data set from internet.

        Either save path or root path must not be None and not both.

        Returns Xtr, Ytr, Xte, Yte as numpy arrays
    '''

    assert ((save_dir is not None and root_path is None) or (save_dir is None and root_path is not None))

    if root_path is None:
        print('Downloading STL-10 dataset...')
        data_url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
        filename = data_url.split('/')[-1]
        filepath = os.path.join(save_dir, filename)
        filepath, _ = urllib.request.urlretrieve(data_url, filepath)
        print('Download Done, Extracting... [%s]' % filename)
        tarfile.open(filepath, 'r:gz').extractall(save_dir)

    if root_path is None:
        x_train, y_train, x_test, y_test = load_stl_10(os.path.join(save_dir, "stl10_binary"))
    else:
        x_train, y_train, x_test, y_test = load_stl_10(os.path.join(root_path, "stl10_binary"))

    # resize images
    x_train = np.array([cv2.resize(img, dsize=(32, 32)) for img in x_train])
    x_test = np.array([cv2.resize(img, dsize=(32, 32)) for img in x_test])

    # format dataset to channels x height x width
    x_train = np.rollaxis(x_train, 3, 1)
    x_test = np.rollaxis(x_test, 3, 1)

    print('Xtrain shape', x_train.shape)
    print('Ytrain shape', y_train.shape)
    print('Xtest shape', x_test.shape)
    print('Ytest shape', y_test.shape)

    return x_train, y_train, x_test, y_test
