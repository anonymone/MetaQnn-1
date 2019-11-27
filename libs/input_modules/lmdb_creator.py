import argparse
import caffe
import lmdb
import numpy as np
import os
import shutil

import get_datasets
import preprocessing


def add_padding(data, pad, pad_value):
    if pad <= 0:
        return data
    print("Adding pads [%d]" % pad)
    shp = data.shape
    padded = np.zeros((shp[0], shp[1], shp[2] + 2 * pad, shp[3] + 2 * pad))
    padded[:, :, :, :] = pad_value
    padded[:, :, pad:-pad, pad:-pad] = data
    print(padded.shape)
    return padded


def create_record(X, y, path, save_as_float=False):
    '''Creates a single LMDB file. Path is the full path of filename to store'''

    assert (X.shape[0] == y.shape[0])
    assert (y.min() == 0)
    if os.path.isdir(path):
        print('removing ' + path)
        shutil.rmtree(path)

    N = X.shape[0]

    map_size = X.nbytes * 50

    env = lmdb.open(path, map_size=map_size)

    print('creating ' + path)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            if save_as_float:
                datum.float_data.extend(X[i].astype(float).flat)
            else:
                datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(y[i])
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


def shuffle(X, y):
    ''' X and y must be the same length vectors '''
    assert X.shape[0] == y.shape[0]
    assert len(X.shape) == 4
    new_index = np.random.permutation(np.arange(X.shape[0]))
    return X[new_index, :, :, :], y[new_index]


def create_records(x_train,
                   y_train,
                   x_test,
                   y_test,
                   root_path,
                   number_val=0,
                   per_image_fn=None,
                   gcn=False,
                   mean_subtraction=False,
                   save_as_float=False,
                   pad=0,
                   x_validation=None,
                   y_validation=None):
    ''' Splits x_train in validation and train sets. Also saves a full train lmdb.
        full train and split train are both first shuffled before being saved

        If both x_validation is not None AND number_val > 0, we create a new validation set from x_train combined with x_validation
    '''
    print('Labels train', np.unique(y_train))
    print('Labels test', np.unique(y_test))

    if save_as_float:
        print('Converting to Float')
        x_train = x_train.astype(float)
        y_train = y_train.astype(float)
        x_test = x_test.astype(float)
        y_test = y_test.astype(float)
        if x_validation is not None:
            x_validation = x_validation.astype(float)
            y_validation = y_validation.astype(float)

    if per_image_fn is not None:
        print('Applying ' + per_image_fn.__name__ + ' to training set')
        for i in range(x_train.shape[0]):
            x_train[i] = per_image_fn(x_train[i].T).T
        print('Applying ' + per_image_fn.__name__ + ' to testing set')
        for i in range(x_test.shape[0]):
            x_test[i] = per_image_fn(x_test[i].T).T
        if x_validation is not None:
            print('Applying ' + per_image_fn.__name__ + ' to validation set')
            for i in range(x_validation.shape[0]):
                x_validation[i] = per_image_fn(x_validation[i].T).T

    if x_validation is not None:
        train_x = x_train.copy()
        train_y = y_train.copy()
        val_x = x_validation.copy()
        val_y = y_validation.copy()
        x_train = np.concatenate([x_train, x_validation])
        y_train = np.concatenate([y_train, y_validation])

    if number_val:
        train_x = []
        train_y = []
        val_x = []
        val_y = []
        for i in np.unique(y_train):
            X_label = x_train[y_train == i].copy()
            proportion = float(X_label.shape[0]) / x_train.shape[0]
            divide = int(round(proportion * number_val))

            # Deal with wierd rounding error
            if val_x and np.concatenate(val_x).shape[0] + divide > number_val:
                divide = number_val - np.concatenate(val_x).shape[0]

            val_x.append(X_label[:divide])
            val_y.append([i] * divide)
            train_x.append(X_label[divide:])
            train_y.append([i] * (X_label.shape[0] - divide))

        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        val_x = np.concatenate(val_x)
        val_y = np.concatenate(val_y)
        assert (val_x.shape[0] == number_val)

    if x_validation is not None or number_val > 0:

        train_x, train_y = shuffle(train_x, train_y)

        if gcn:
            print('Train Small Before GCN Mean, std ', np.mean(train_x), np.std(train_x))
            print('Validation Before GCN mean, std', np.mean(val_x), np.std(val_x))
            train_x, val_x = preprocessing.gcn_whiten(train_x, val_x)
            print('Train Small Mean, std: ', np.mean(train_x), np.std(train_x))
            print('Validation Mean, std: ', np.mean(val_x), np.std(val_x))

        if mean_subtraction:
            train_x, val_x = preprocessing.mean_subtraction(train_x, val_x)
            print('Train Small Mean: ', np.mean(train_x))
            print('Validation Mean: ', np.mean(train_x))

        if pad:
            pad_value = 0 if gcn or mean_subtraction else 128
            train_x = add_padding(train_x, pad, pad_value)

        print('Train Small x shape', train_x.shape, train_x.dtype)
        print('Train Small y shape', train_y.shape, train_y.dtype)
        print('Validation x shape', val_x.shape, val_x.dtype)
        print('Validation y shape', val_y.shape, val_y.dtype)
        print('Biggest Class is %f of training set' % (
                np.unique(train_y, return_counts=True)[1].max() / float(len(train_y))))
        print('Biggest Class is %f of validation set' % (
                np.unique(val_y, return_counts=True)[1].max() / float(len(val_y))))

        create_record(train_x, train_y, os.path.join(root_path, 'train.lmdb'), save_as_float=save_as_float)
        create_record(val_x, val_y, os.path.join(root_path, 'val.lmdb'), save_as_float=save_as_float)
        del train_x, train_y, val_x, val_y

    x_train, y_train = shuffle(x_train, y_train)

    if gcn:
        print('Train Small Before GCN Mean, std ', np.mean(x_train), np.std(x_train))
        print('Test Before GCN mean, std', np.mean(x_test), np.std(x_test))
        x_train, x_test = preprocessing.gcn_whiten(x_train, x_test)
        print('Train Mean, std: ', np.mean(x_train), np.std(x_train))
        print('Test Mean, std: ', np.mean(x_test), np.std(x_test))

    if mean_subtraction:
        x_train, x_test = preprocessing.mean_subtraction(x_train, x_test)
        print('Train Mean: ', np.mean(x_train))
        print('Test Mean: ', np.mean(x_test))

    if pad:
        pad_value = 0 if gcn or mean_subtraction else 128
        x_train = add_padding(x_train, pad, pad_value)

    print('Train x shape', x_train.shape, x_train.dtype)
    print('Train y shape', y_train.shape, y_train.dtype)
    print('Test x shape', x_test.shape, x_test.dtype)
    print('Test y shape', y_test.shape, y_test.dtype)

    create_record(x_train, y_train, os.path.join(root_path, 'train_full.lmdb'), save_as_float=save_as_float)
    create_record(x_test, y_test, os.path.join(root_path, 'test.lmdb'), save_as_float=save_as_float)


def main():
    parser = argparse.ArgumentParser()
    dataset_options = ['caltech101', 'cifar10', 'cifar100', 'svhn', 'svhn_full', 'svhn_small', 'mnist', 'fashion_mnist',
                       'flowers102', 'food101', 'flower5', 'stl10']
    parser.add_argument('dataset', choices=dataset_options, help='Which data set')
    parser.add_argument('-v', '--number_val', help='How many validation images', type=int, default=0)
    parser.add_argument('-prep', '--preprocessing', help='Which per image preprocessing function to use', default=None,
                        choices=['lcn', 'standard_whiten'])
    parser.add_argument('-gcn', '--gcn', help='Whether to use global contrast normalization or not. Default is false',
                        default=False, type=bool)
    parser.add_argument('-ms', '--mean_subtraction', help='Do global mean subtraction?', default=False, type=bool)
    parser.add_argument('-pad', '--padding', help='Padding value on each side.', type=int, default=0)

    args = parser.parse_args()
    lmdb_directory = os.path.abspath('./' + args.dataset)
    if not os.path.isdir(lmdb_directory):
        os.makedirs(lmdb_directory)
    dataset_directory = os.path.join(lmdb_directory, 'dataset')

    # Should we save as float?
    save_as_float = args.preprocessing is not 'none' or args.gcn or args.mean_subtraction

    # Get preprocessing function
    if args.preprocessing == 'lcn':
        per_image_fn = preprocessing.lcn_whiten
    elif args.preprocessing == 'standard_whiten':
        per_image_fn = preprocessing.standard_whiten
    else:
        per_image_fn = None

    padding = args.padding if args.padding else 0

    if args.dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_datasets.get_cifar10(save_dir=dataset_directory)
        x_validation, y_validation = None, None
    elif args.dataset == 'cifar100':
        x_train, y_train, x_test, y_test = get_datasets.get_cifar100(save_dir=dataset_directory)
        x_validation, y_validation = None, None
    elif args.dataset == 'svhn':
        x_train, y_train, x_test, y_test = get_datasets.get_svhn(save_dir=dataset_directory)
        x_validation, y_validation = None, None
    elif args.dataset == 'svhn_full':
        x_train, y_train, x_validation, y_validation, x_test, y_test = get_datasets.get_svhn_full(
            save_dir=dataset_directory)
    elif args.dataset == 'svhn_small':
        x_train, y_train, x_validation, y_validation, x_test, y_test = get_datasets.get_svhn_small(
            save_dir=dataset_directory)
    elif args.dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_datasets.get_mnist(save_dir=dataset_directory)
        x_validation, y_validation = None, None
    elif args.dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_datasets.get_fashion_mnist(save_dir=dataset_directory)
        x_validation, y_validation = None, None
    elif args.dataset == 'caltech101':
        x_train, y_train, x_validation, y_validation = get_datasets.get_caltech101(save_dir=dataset_directory)
        x_test, y_test = x_validation.copy(), y_validation.copy()
    elif args.dataset == 'flowers102':
        x_train, y_train, x_validation, y_validation, x_test, y_test = get_datasets.get_flowers_102(
            save_dir=dataset_directory)
    elif args.dataset == 'flower5':
        x_train, y_train, x_validation, y_validation, x_test, y_test = get_datasets.get_flowers_5(
            save_dir=dataset_directory)
    elif args.dataset == 'food101':
        x_train, y_train, x_validation, y_validation, x_test, y_test = get_datasets.get_food_101(
            save_dir=dataset_directory)
    elif args.dataset == 'stl10':
        x_train, y_train, x_test, y_test = get_datasets.get_stl_10(save_dir=dataset_directory)
        x_validation, y_validation = None, None

    create_records(x_train=x_train,
                   y_train=y_train,
                   x_test=x_test,
                   y_test=y_test,
                   root_path=lmdb_directory,
                   number_val=args.number_val,
                   per_image_fn=per_image_fn,
                   gcn=args.gcn,
                   mean_subtraction=args.mean_subtraction,
                   save_as_float=save_as_float,
                   pad=padding,
                   x_validation=x_validation,
                   y_validation=y_validation)


if __name__ == "__main__":
    main()

