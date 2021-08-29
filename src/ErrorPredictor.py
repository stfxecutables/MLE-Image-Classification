import logging

import numpy as np
from sklearn.neural_network import MLPClassifier as MLP


class ErrorPredictor:
    """
    Class to detect and correct errors in aggregate predictions using ensemble prediction-matrices.
    NOTE: Current implementation is only for binary (0, 1) classification
    """

    logger = logging.getLogger(__name__)

    def __init__(self):
        """
        Class constructor. Initialized MLP model
        """
        self.model = MLP(
            hidden_layer_sizes=(500, 500, 250, 100, 200),
            activation="relu",
            solver="adam",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains MLP to detect errors on aggregate predictions with prediction matrices from ensembles
        :param X: numpy array of prediction matrices from ensemble
        :param y: numpy array of labels corresponding to each prediction matrix
        :return: None
        """
        logging.info("Starting training for ErrorPredictor")
        train_X = self.format_prediction_matrix(X)
        incorrect_predictions = self.get_incorrect_predictions(X, y)
        self.model.fit(train_X, incorrect_predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses prediction matrix to create initial predictions using aggregate scores and flips prediction detected
        incorrect by MLP model
        :param X: numpy array of prediction matrices from ensemble
        :return: 1D numpy array containing final predictions
        """
        test_X = self.format_prediction_matrix(X)
        initial_predictions = np.argmax(np.sum(X, axis=0), axis=-1)
        incorrect_predictions = self.model.predict(test_X)

        corrected_predictions = np.copy(initial_predictions)
        corrected_predictions[incorrect_predictions] = np.abs(
            corrected_predictions[incorrect_predictions] - 1
        )

        return corrected_predictions

    @staticmethod
    def format_prediction_matrix(ensemble_predictions: np.ndarray) -> np.ndarray:
        """
        Appends prediction sums to prediction matrices and flattens prediction matrices to 1D arrays for MLP
        :param ensemble_predictions: array of prediction matrices from ensemble
        :return: Reshaped data for input for MLP model
        """
        predictionSums = np.sum(ensemble_predictions, axis=0)

        ensemble_predictions = np.vstack(
            (ensemble_predictions, np.expand_dims(predictionSums, 0))
        )
        data = np.reshape(
            np.swapaxes(ensemble_predictions, 0, 1),
            (np.shape(ensemble_predictions)[1], -1),
        )

        return data

    @staticmethod
    def get_incorrect_predictions(
        ensemble_predictions: np.ndarray, true_labels: np.ndarray
    ) -> np.ndarray:
        """
        :param ensemble_predictions: array of prediction matrices from ensemble
        :param true_labels: numpy array of true labels corresponding to each prediction matrix in ensemble_predictions
        :return: 1D numpy array with incorrect predictions (calculated by aggregation) marked as True (1), and correct
        as False (0)
        """
        assert np.shape(ensemble_predictions)[1] == np.shape(true_labels)[0]

        predictions = np.argmax(np.sum(ensemble_predictions, axis=0), axis=-1)
        return np.not_equal(predictions, true_labels)
