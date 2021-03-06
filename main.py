import configparser
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from src.utils.base_models import get_architecture
from src.utils.get_data import get_dataset

LOGGING_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)20s()] - %(message)s"

# Classes to use to create subset for binary prediction
BINARY_DATASET_CLASSES = {
    "MNIST_BIN": [4, 9],
    "F_MNIST_BIN": [0, 6],  # T-Shirt and Shirt
    "CIFAR10_BIN": [3, 5],  # Dog and Cat
}


def format_data(X: np.ndarray, y: np.ndarray) -> Tuple:
    """
    Formats data for training base-learners in ensemble
    :param X: numpy array containing images
    :param y: numpy array containing class labels corresponding to images
    :return: Tuple containing data with channels and one-hot encoded labels
    """
    assert 3 <= X.ndim <= 4, "Image dimensions not supported"

    from tensorflow.keras.utils import to_categorical

    # if image data doesn't have channels, create a fake - greyscale channel
    if X.ndim == 3:
        X = np.expand_dims(X, -1).astype(np.float32)
        logging.info(f"Channel added for image data for {DATASET}")
    y = to_categorical(y, num_classes=np.shape(np.unique(y))[0])

    return X.astype("float16"), y


def ga_predict(
    prediction_matrix_s2: np.ndarray,
    labels_s2: np.ndarray,
    prediction_matrix_test: np.ndarray,
) -> None:
    """
    Uses GA class object to create and optimize a weight matrix for weighted-aggregated predictions. Final predictions
    are saved to disk in ROOT_DIR with name ga_test_predictions.npy
    :param prediction_matrix_s2: prediction matrix generated by ensemble on stage2 training / validation data
    :param labels_s2: true labels corresponding to each prediction matrix in prediction_matrix_s2
    :param prediction_matrix_test: prediction matrix generated on testing data
    :return: None
    """
    from src.GA import GA

    weight_matrix_optimizer = GA(prediction_matrix_s2, labels_s2, GA_POPULATION)
    for generation_num in range(GA_GENERATIONS):
        weight_matrix_optimizer.next_gen()

    test_predictions = weight_matrix_optimizer.predict(prediction_matrix_test)
    np.save(str(ROOT_DIR.joinpath("ga_test_predictions")), test_predictions)
    logging.info(
        f"GA generated final testing predictions stored at {ROOT_DIR.joinpath('ga_test_predictions.npy')}"
    )


def cnn_predict(
    prediction_matrix_s2: np.ndarray,
    labels_s2: np.ndarray,
    prediction_matrix_test: np.ndarray,
) -> None:
    """
    Uses CNNAnswerPredictor class object to act as a super-learner and generate predictions from prediction matrices.
    Final predictions are saved to disk in ROOT_DIR with name cnn_test_predictions.npy
    :param prediction_matrix_s2: prediction matrix generated by ensemble on stage2 training / validation data
    :param labels_s2: true labels corresponding to each prediction matrix in prediction_matrix_s2
    :param prediction_matrix_test: prediction matrix generated on testing data
    :return: None
    """
    from src.CNN_Answer_Predictor import CNNAnswerPredictor

    super_learner = CNNAnswerPredictor(NUM_BASE_LEARNERS, UNITS)
    super_learner.train(prediction_matrix_s2, labels_s2)

    test_predictions = super_learner.predict(prediction_matrix_test)
    np.save(str(ROOT_DIR.joinpath("cnn_test_predictions")), test_predictions)
    logging.info(
        f"CNN generated final testing predictions stored at {ROOT_DIR.joinpath('cnn_test_predictions.npy')}"
    )


def error_correction(
    prediction_matrix_s2: np.ndarray,
    labels_s2: np.ndarray,
    prediction_matrix_test: np.ndarray,
) -> None:
    """
    Uses ErrorPredictor to detect and correct for errors in aggregate predictions for binary classification
    :param prediction_matrix_s2: prediction matrix generated by ensemble on stage2 training / validation data
    :param labels_s2: true labels corresponding to each prediction matrix in prediction_matrix_s2
    :param prediction_matrix_test: prediction matrix generated on testing data
    :return: None
    """
    from src.ErrorPredictor import ErrorPredictor

    error_predictor = ErrorPredictor()
    error_predictor.fit(prediction_matrix_s2, labels_s2)

    test_predictions = error_predictor.predict(prediction_matrix_test)
    np.save(str(ROOT_DIR.joinpath("ec_test_predictions")), test_predictions)
    logging.info(
        f"Error corrected final testing predictions stored at {ROOT_DIR.joinpath('ec_test_predictions.npy')}"
    )


def generate_ensemble():
    """
    Creates and trains ensemble based on the config settings specified. Ensemble models are saved to disk in
    ROOT_DIR/models. Prediction matrices for all data splits are saved
    :return:
    """
    from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
    from src.Ensemble import Ensemble

    tr1_X, tr1_y = format_data(train_X, train_y)
    tr2_X, tr2_y = format_data(val_X, val_y)
    tst_X, tst_y = format_data(test_X, test_y)

    if ENSEMBLE_TYPE == "full":
        tr1_X = np.vstack((tr1_X, tr2_X))
        tr1_y = np.vstack((tr1_y, tr2_y))

    base_architecture = get_architecture(DATASET, UNITS)
    ensemble = Ensemble.from_arch(
        architecture=base_architecture, num_models=NUM_BASE_LEARNERS
    )
    ensemble.compile(
        optimizer="adam", threshold=THRESHOLD, loss=LOSS_TYPE, metrics=["acc"]
    )

    ensemble.fit(
        tr1_X,
        tr1_y,
        epochs=30,
        callbacks=[
            ReduceLROnPlateau(),
            TerminateOnNaN(),
            EarlyStopping(
                baseline=0.85, patience=15, monitor="val_acc", restore_best_weights=True
            ),
        ],
    )

    # Save ensemble models
    ensemble.save_models(ROOT_DIR)

    # Save predictions for all data splits
    tr1_predictions = ensemble.predict(tr1_X)
    np.save(str(ROOT_DIR.joinpath("training_prediction_matrices")), tr1_predictions)
    logging.info(
        f"Training predictions from ensemble saved at {ROOT_DIR.joinpath('training_prediction_matrices.npy')}"
    )

    if ENSEMBLE_TYPE != "full":
        tr2_predictions = ensemble.predict(tr2_X)
        np.save(str(ROOT_DIR.joinpath("val_prediction_matrices")), tr2_predictions)
        logging.info(
            f"Validation/Training phase 2 predictions from ensemble saved at "
            f"{ROOT_DIR.joinpath('val_prediction_matrices.npy')}"
        )

    test_predictions = ensemble.predict(tst_X)
    np.save(str(ROOT_DIR.joinpath("test_prediction_matrices")), test_predictions)
    logging.info(
        f"Testing predictions from ensemble saved at {ROOT_DIR.joinpath('test_prediction_matrices.npy')}"
    )


if __name__ == "__main__":
    tf.get_logger().setLevel(logging.ERROR)
    logging.basicConfig(
        format=LOGGING_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
    )

    config_parser = configparser.ConfigParser()
    config_parser.read("default.ini")
    config_parser.read("MLEIC.ini")

    # CONFIG SETTINGS FOR DATASET
    DATASET = config_parser["General"].get("Dataset")
    VALIDATION_PROPORTION = config_parser["General"].getfloat("ValidationProportion")
    logging.info(
        f"Dataset config: DATASET={DATASET}, VALIDATION_PROPORTION={VALIDATION_PROPORTION}"
    )

    # CONFIG SETTINGS FOR ENSEMBLE
    ENSEMBLE_TYPE = config_parser["Ensemble"].get("Type")
    NUM_BASE_LEARNERS = config_parser["Ensemble"].getint("NumBaseLearners")
    LOSS_TYPE = "categorical_crossentropy"
    THRESHOLD = 0.0
    if config_parser["Ensemble"].getboolean("UseDynamicLoss"):
        LOSS_TYPE = "custom"
        THRESHOLD = config_parser["Ensemble"].getfloat("DynamicLossThreshold")
    logging.info(
        f"Ensemble config: ENSEMBLE_TYPE={ENSEMBLE_TYPE}, NUM_BASE_LEARNERS={NUM_BASE_LEARNERS}, "
        f"LOSS_TYPE={LOSS_TYPE}, THRESHOLD={THRESHOLD}"
    )

    # CONFIG SETTINGS FOR GA
    GA_GENERATIONS = config_parser["GA"].getint("GAGenerations")
    GA_POPULATION = config_parser["GA"].getint("GAPopulation")
    logging.info(
        f"GA config: GA_GENERATIONS={GA_GENERATIONS}, GA_POPULATION={GA_POPULATION}"
    )

    # Check if dataset if MNIST_BIN, F_MNIST_BIN, or CIFAR10_BIN
    UNITS = 10
    if DATASET in BINARY_DATASET_CLASSES:
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = get_dataset(
            DATASET, VALIDATION_PROPORTION, BINARY_DATASET_CLASSES[DATASET]
        )
        UNITS = 2
    else:
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = get_dataset(
            DATASET, VALIDATION_PROPORTION
        )

    # Set root directory based on configured settings
    if LOSS_TYPE == "custom":
        ROOT_DIR = Path(
            f"./models/{DATASET}/dynamic_loss/{ENSEMBLE_TYPE}/threshold_{THRESHOLD}"
        )
    else:
        ROOT_DIR = Path(f"./models/{DATASET}/categorical_crossentropy/{ENSEMBLE_TYPE}")
    logging.info(f"ROOT_DIR for ensemble model: {ROOT_DIR}")

    # Generate ensemble and prediction matrices
    generate_ensemble()

    # Stage 2 training (if model type is not full)
    if ENSEMBLE_TYPE != "full":
        training_prediction_mat = np.load(
            str(ROOT_DIR.joinpath("training_prediction_matrices.npy"))
        )
        validation_prediction_mat = np.load(
            str(ROOT_DIR.joinpath("val_prediction_matrices.npy"))
        )
        testing_prediction_mat = np.load(
            str(ROOT_DIR.joinpath("test_prediction_matrices.npy"))
        )

        s2_X = np.concatenate(
            (training_prediction_mat, validation_prediction_mat), axis=1
        )
        s2_y = np.concatenate((train_y, val_y), axis=-1)

        ga_predict(s2_X, s2_y, testing_prediction_mat)
        cnn_predict(s2_X, s2_y, testing_prediction_mat)

        # Error prediction and correction if dataset in use is a binary dataset
        if DATASET in BINARY_DATASET_CLASSES:
            error_correction(s2_X, s2_y, testing_prediction_mat)
