from pathlib import Path
from typing import List, Tuple, Union

import idx2numpy
import numpy as np
from sklearn.model_selection import train_test_split


def read_saved(data_dir: Path) -> Tuple:
    """
    reads saved - split (training, validation/train phase 2, test) data from disk
    :param data_dir: Path object containing the path to the data directory
    :return: tuple of training set, validation/training phase 2 set, testing set.
    "set" here is a tuple of training and testing numpy arrays
    """
    train_data = np.load(str(data_dir.joinpath("train_data.npy"))).astype("float16")
    train_labels = np.load(str(data_dir.joinpath("train_labels.npy")))
    val_data = np.load(str(data_dir.joinpath("val_data.npy"))).astype("float16")
    val_labels = np.load(str(data_dir.joinpath("val_labels.npy")))
    test_data = np.load(str(data_dir.joinpath("test_data.npy"))).astype("float16")
    test_labels = np.load(str(data_dir.joinpath("test_labels.npy")))

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def write_data(
    data_dir: Path,
    train_data,
    train_labels,
    val_data,
    val_labels,
    test_data,
    test_labels,
) -> None:
    """
    writes data splits to disk. Data and labels will be written as numpy files (.npy)
    :param data_dir: Path object containing location of directory to save data and labels in
    :param train_data: list or numpy array containing training data
    :param train_labels: list or numpy array containing training labels
    :param val_data: list or numpy array containing validation/training phase 2 data
    :param val_labels: list or numpy array containing validation/training phase 2 labels
    :param test_data: list or numpy array containing testing data
    :param test_labels: list or numpy array containing testing labels
    :return: None
    """
    np.save(str(data_dir.joinpath("train_data.npy")), train_data)
    np.save(str(data_dir.joinpath("train_labels.npy")), train_labels)
    np.save(str(data_dir.joinpath("val_data.npy")), val_data)
    np.save(str(data_dir.joinpath("val_labels.npy")), val_labels)
    np.save(str(data_dir.joinpath("test_data.npy")), test_data)
    np.save(str(data_dir.joinpath("test_labels.npy")), test_labels)


def get_subset(
    classes: List,
    train_data,
    train_labels,
    val_data,
    val_labels,
    test_data,
    test_labels,
) -> Tuple:
    """
    creates a subset of training, validation, and testing set using the specified list of classes to select
    :param classes: list of classes in the labels that are to be selected in the subset
    :param train_data: list or numpy array containing training data
    :param train_labels: list or numpy array containing training labels
    :param val_data: list or numpy array containing validation/training phase 2 data
    :param val_labels: list or numpy array containing validation/training phase 2 labels
    :param test_data: list or numpy array containing testing data
    :param test_labels: list or numpy array containing testing labels
    :return: tuple of training sub-set, validation/training phase 2 sub-set, testing sub-set.
    "sub-set" here is a tuple of training and testing numpy arrays
    """
    train_set = np.isin(train_labels, classes)
    val_set = np.isin(val_labels, classes)
    test_set = np.isin(test_labels, classes)

    train_data = train_data[train_set]
    train_labels = train_labels[train_set] == train_set[0]
    val_data = val_data[val_set]
    val_labels = val_labels[val_set] == train_set[0]
    test_data = test_data[test_set]
    test_labels = test_labels[test_set] == train_set[0]

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def get_dataset(
    dataset: str, validation_proportion: float = 0.0, classes: List = None
) -> Union[Tuple, None]:
    """
    gets dataset based on dataset string name specified
    :param dataset: string representing dataset in use
    :param validation_proportion: float between 0.0-1.0 to specify proportion of original training set to use for
    creating validation split
    :param classes: List of classes from the original set to use to create subset (if this is not specified, all classes
    will be used)
    :return: tuple of training set, validation/training phase 2 set, testing set.
    "set" here is a tuple of training and testing numpy arrays
    """
    if dataset == "MNIST" or dataset == "MNIST_BIN":
        return get_mnist(validation_proportion, classes)
    elif dataset == "F_MNIST" or dataset == "F_MNIST_BIN":
        return get_fashion_mnist(validation_proportion, classes)
    elif dataset == "CIFAR10" or dataset == "CIFAR10_BIN":
        return get_cifar10(validation_proportion, classes)
    return None


def get_fashion_mnist(
    validation_proportion: float = 0.0, classes: List = None
) -> Tuple:
    """
    gets fashion MNIST data from disk. If previous split for given validation proportion is found, it will be reused
    :param validation_proportion: float between 0.0-1.0 to specify proportion of original training set to use for
    creating validation split
    :param classes: List of classes from the original set to use to create subset (if this is not specified, all classes
    will be used)
    :return: tuple of training set, validation/training phase 2 set, testing set.
    "set" here is a tuple of training and testing numpy arrays
    """
    assert (
        0.0 <= validation_proportion <= 1.0
    ), "validation_proportion should be in range [0.0, 1.0]"
    split_dir = Path("data/splits/f-mnist-" + str(validation_proportion))

    if Path.is_dir(split_dir):
        (
            (train_data, train_labels),
            (val_data, val_labels),
            (test_data, test_labels),
        ) = read_saved(split_dir)
        if classes is not None:
            return get_subset(
                classes,
                train_data,
                train_labels,
                val_data,
                val_labels,
                test_data,
                test_labels,
            )
        return (
            (train_data, train_labels),
            (val_data, val_labels),
            (test_data, test_labels),
        )

    train_data = idx2numpy.convert_from_file(
        str(Path("data/f-mnist/train-images-idx3-ubyte"))
    )
    train_labels = idx2numpy.convert_from_file(
        str(Path("data/f-mnist/train-labels-idx1-ubyte"))
    )

    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=validation_proportion, stratify=train_labels
    )

    test_data = idx2numpy.convert_from_file(
        str(Path("data/f-mnist/t10k-images-idx3-ubyte"))
    )
    test_labels = idx2numpy.convert_from_file(
        str(Path("data/f-mnist/t10k-labels-idx1-ubyte"))
    )

    split_dir.mkdir(parents=True, exist_ok=True)
    write_data(
        split_dir,
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
    )

    if classes is not None:
        return get_subset(
            classes,
            train_data,
            train_labels,
            val_data,
            val_labels,
            test_data,
            test_labels,
        )
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def get_mnist(validation_proportion: float = 0.0, classes: List = None) -> Tuple:
    """
    gets MNIST data from disk. If previous split for given validation proportion is found, it will be reused
    :param validation_proportion: float between 0.0-1.0 to specify proportion of original training set to use for
    creating validation split
    :param classes: List of classes from the original set to use to create subset (if this is not specified, all classes
    will be used)
    :return: tuple of training set, validation/training phase 2 set, testing set.
    "set" here is a tuple of training and testing numpy arrays
    """
    assert (
        0.0 <= validation_proportion <= 1.0
    ), "validation_proportion should be in range [0.0, 1.0]"
    split_dir = Path("data/splits/mnist-" + str(validation_proportion))

    if Path.is_dir(split_dir):
        (
            (train_data, train_labels),
            (val_data, val_labels),
            (test_data, test_labels),
        ) = read_saved(split_dir)
        if classes is not None:
            return get_subset(
                classes,
                train_data,
                train_labels,
                val_data,
                val_labels,
                test_data,
                test_labels,
            )
        return (
            (train_data, train_labels),
            (val_data, val_labels),
            (test_data, test_labels),
        )

    train_data = idx2numpy.convert_from_file(
        str(Path("data/mnist/train-images.idx3-ubyte"))
    )
    train_labels = idx2numpy.convert_from_file(
        str(Path("data/mnist/train-labels.idx1-ubyte"))
    )

    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=validation_proportion, stratify=train_labels
    )

    test_data = idx2numpy.convert_from_file(
        str(Path("data/mnist/t10k-images.idx3-ubyte"))
    )
    test_labels = idx2numpy.convert_from_file(
        str(Path("data/mnist/t10k-labels.idx1-ubyte"))
    )

    split_dir.mkdir(parents=True, exist_ok=True)
    write_data(
        split_dir,
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
    )

    if classes is not None:
        return get_subset(
            classes,
            train_data,
            train_labels,
            val_data,
            val_labels,
            test_data,
            test_labels,
        )
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def get_cifar10(validation_proportion: float = 0.0, classes: List = None) -> Tuple:
    """
    gets CIFAR-10 data from disk. If previous split for given validation proportion is found, it will be reused
    :param validation_proportion: float between 0.0-1.0 to specify proportion of original training set to use for
    creating validation split
    :param classes: List of classes from the original set to use to create subset (if this is not specified, all classes
    will be used)
    :return: tuple of training set, validation/training phase 2 set, testing set.
    "set" here is a tuple of training and testing numpy arrays
    """
    assert (
        0.0 <= validation_proportion <= 1.0
    ), "validation_proportion should be in range [0.0, 1.0]"

    def unpickle(file):
        import pickle

        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    split_dir = Path("data/splits/cifar10-" + str(validation_proportion))

    if Path.is_dir(split_dir):
        (
            (train_data, train_labels),
            (val_data, val_labels),
            (test_data, test_labels),
        ) = read_saved(split_dir)
        if classes is not None:
            return get_subset(
                classes,
                train_data,
                train_labels,
                val_data,
                val_labels,
                test_data,
                test_labels,
            )
        return (
            (train_data, train_labels),
            (val_data, val_labels),
            (test_data, test_labels),
        )

    data_dir = Path("data/cifar-10")
    train_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    train_data = np.empty((50000, 3, 32, 32))
    train_labels = np.empty(50000)
    for i, file in enumerate(train_files):
        this_dict = unpickle(str(data_dir.joinpath(file)))
        this_images = np.reshape(this_dict[b"data"], (10000, 3, 32, 32))
        train_data[i * 10000 : (i + 1) * 10000, :, :, :] = this_images.astype("float16")
        train_labels[i * 10000 : (i + 1) * 10000] = this_dict[b"labels"]

    test_dict = unpickle(str(data_dir.joinpath("test_batch")))
    test_data = np.reshape(test_dict[b"data"], (10000, 3, 32, 32)).astype("float16")
    test_labels = test_dict[b"labels"]

    train_data = np.moveaxis(train_data, 1, -1)
    test_data = np.moveaxis(test_data, 1, -1)

    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=validation_proportion, stratify=train_labels
    )

    split_dir.mkdir()
    write_data(
        split_dir,
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
    )

    if classes is not None:
        return get_subset(
            classes,
            train_data,
            train_labels,
            val_data,
            val_labels,
            test_data,
            test_labels,
        )
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
