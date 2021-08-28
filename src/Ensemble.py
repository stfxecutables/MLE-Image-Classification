import json
import logging
from pathlib import Path
from typing import List, Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import Sequential
from tensorflow.keras import backend as kb
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import save_model, load_model, clone_model


class Ensemble:
    """
    Class to handle ensemble operations - training, predicting, saving, and loading
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        models: List[Sequential],
        use_dynamic_loss: bool = False,
        dynamic_loss_threshold: float = 0.0,
    ):
        """
        Class initializer. Sets ensemble models to list of models given
        :param models: List of Keras Sequential models
        :param use_dynamic_loss: boolean value specifying whether or not to use dynamic loss to train the models
        :param dynamic_loss_threshold: float in range [0.0, 1.0] to use if models are to train using dynamic loss
        """
        self.models = models
        num_models = len(models)

        if use_dynamic_loss:
            assert (
                0.0 <= dynamic_loss_threshold <= 1.0
            ), "Dynamic loss threshold must be in range [0.0, 1.0]"
            if dynamic_loss_threshold < 0.6:
                self.logger.warning(
                    f"Dynamic loss threshold ({dynamic_loss_threshold}) < recommended value (0.6)"
                )

        self.meta_data = {
            "num_models": num_models,
            "use_dynamic_loss": use_dynamic_loss,
            "threshold": dynamic_loss_threshold,
        }

    @classmethod
    def from_dir(cls, directory: Path):
        """
        Loads ensemble object and meta-data from directory
        :param directory: Path object containing path to directory that has saved ensemble files
        :return: Ensemble object
        """
        assert Path.is_dir(directory), "Directory to load ensemble not found"

        assert Path.is_file(
            directory.joinpath("meta_data.txt")
        ), "Meta data file for ensemble not found"
        with open(directory.joinpath("meta_data.txt"), "r") as meta_file:
            meta_data = json.loads(meta_file.readline())

        models: List[Sequential] = []
        for i in range(meta_data["num_models"]):
            cls.logger.info(
                "Loading model {model_num} from {path}".format(
                    model_num=i, path=directory
                )
            )
            if meta_data["use_dynamic_loss"]:
                this_model = load_model(
                    directory.joinpath("model_" + str(i)),
                    custom_objects={
                        "dynamic_loss": cls.dynamic_loss_wrapper(meta_data["threshold"])
                    },
                )
            else:
                this_model = load_model(
                    directory.joinpath("model_" + str(i)),
                )
            models.append(this_model)

        ens = cls(models)
        ens.meta_data = meta_data

        return ens

    @classmethod
    def from_arch(
        cls,
        architecture: Sequential,
        num_models: int,
        use_dynamic_loss: bool = False,
        dynamic_loss_threshold: float = 0.0,
    ):
        """
        Creates ensemble object using given base model
        :param architecture: Keras Sequential object to use for all base models
        :param num_models: number of clones of base architecture to make for ensemble
        :param use_dynamic_loss: boolean value specifying whether or not to use dynamic loss to train the models
        :param dynamic_loss_threshold: float in range [0.0, 1.0] to use if models are to train using dynamic loss
        :return: Ensemble object
        """
        assert num_models > 0, "Must have at least 1 model in Ensemble"
        models = [clone_model(architecture) for _ in range(num_models)]
        cls.logger.info(f"Created Ensemble class object with {num_models} models")

        return cls(models, use_dynamic_loss, dynamic_loss_threshold)

    def compile(
        self, optimizer: str, loss: Any, metrics: Any, threshold: float = 0.0
    ) -> None:
        """
        Compiles all the base learners using optimizer, loss, metrics, and threshold specified
        :param optimizer: optimizer to use to train base-learners
        :param loss: loss function to use to train base-learners. Use "custom" to use dynamic-loss
        :param metrics: metrics to use to train base-learners
        :param threshold: threshold to use for training using dynamic-loss. Must be in range [0.0, 1.0]
        :return: None
        """
        if loss == "custom":
            loss = self.dynamic_loss_wrapper(threshold=threshold)
            self.meta_data["use_dynamic_loss"] = True
            self.meta_data["threshold"] = threshold

        for model_num, model in enumerate(self.models):
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            self.logger.info(f"Compiled {model_num} in Ensemble object")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, **kwargs) -> None:
        """
        Trains all base-learners in the ensemble on the given data and labels for number of epochs specified
        :param X: numpy array containing training data
        :param y: numpy array containing training labels corresponding to training data
        :param epochs: number of epochs to train each base-learner for
        :param kwargs: additional key-word arguments supported by tensorflow.keras.Sequential.fit
        :return: None
        """
        assert (
            np.shape(X)[0] == np.shape(y)[0]
        ), "Number of samples ({}) must match number of labels ({})".format(
            np.shape(X)[0], np.shape(y)[0]
        )

        data_splits = StratifiedShuffleSplit(
            n_splits=self.meta_data["num_models"]
        ).split(X, y)

        for model_num, (train_idx, test_idx) in enumerate(data_splits):
            train_X, test_X = X[train_idx, :, :], X[test_idx, :, :]
            train_y, test_y = y[train_idx], y[test_idx]

            self.logger.info(f"Starting training for model {model_num} in Ensemble")
            self.models[0].fit(
                x=train_X,
                y=train_y,
                validation_data=(test_X, test_y),
                epochs=epochs,
                **kwargs,
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions on given data from all base-learners
        :param X: numpy array containing data to make predictions on
        :return: numpy array containing predictions. Shape: (# base-learners, # samples, # classes)
        """
        predictions: List[np.ndarray] = []
        for model in self.models:
            predictions.append(model.predict(X))

        return np.array(predictions)

    def save_models(self, directory: Path) -> None:
        """
        Saves all base-learners along with meta-data to directory
        :param directory: Path object containing path to directory where models and meta data are to be saved
        :return: None
        """
        if not Path.is_dir(directory):
            self.logger.warning(
                "Directory not found for ensemble. Creating directory to store models"
            )
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.logger.error(
                    f"Could not create directory to save ensemble. Check path {directory}"
                )

                # save ensemble in temp directory
                temp_ensemble_num = 1
                directory = Path(f"./temp/temp_ensemble_{temp_ensemble_num}")
                while directory.is_dir():
                    temp_ensemble_num += 1
                    directory = Path(f"./temp/temp_ensemble_{temp_ensemble_num}")
                self.logger.info(f"Saving ensemble to temp directory: {directory}")

        with open(directory.joinpath("meta_data.txt"), "w") as meta_file:
            meta_file.write(json.dumps(self.meta_data))
            self.logger.info(
                f"Meta-data for ensemble saved at {directory.joinpath('meta_data.txt')}"
            )

        for model_num, model in enumerate(self.models):
            save_model(model, directory.joinpath("model_" + str(model_num)))
            self.logger.info(f"Saved model {model_num} in {directory}")

    @staticmethod
    def dynamic_loss_wrapper(threshold: float = 0.6):
        """
        Wrapper function to return dynamic-loss function with specified threshold.
        :param threshold: float in range [0.0, 1.0] above which prediction is considered reliable/correct
        :return: dynamic-loss function with specified threshold
        """

        def dynamic_loss(y_true, y_pred):
            training_threshold = kb.constant(threshold, dtype=kb.floatx())

            # Cast to 1 for every value above trainingThreshold and 0 otherwise
            above_threshold = kb.cast(
                kb.greater(y_pred, training_threshold), kb.floatx()
            )
            # Cast to 0.01 * value for every value below threshold and 0 otherwise
            below_threshold = (
                y_pred
                * kb.cast(kb.less_equal(y_pred, training_threshold), kb.floatx())
                * 0.01
            )
            # Add two tensor matrices calculated above
            new_pred = above_threshold + below_threshold

            # Return value of categorical cross-entropy for modified predictions
            return categorical_crossentropy(y_true, new_pred)

        return dynamic_loss
