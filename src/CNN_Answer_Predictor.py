import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv1D


class CNNAnswerPredictor:
    """
    Class to handle CNN super-learner for Ensemble objects"
    """

    def __init__(self, num_models: int, num_classes: int):
        """
        Class constructor. Initialized a keras model with given num_classes and num_models parameters
        :param num_models: number of models in ensemble
        :param num_classes: number of classes ensemble predicts
        """
        self.nun_models = num_models
        self.num_classes = num_classes
        self.model: keras.Model = self.init_model()

    def init_model(self) -> keras.Model:
        """
        Creates a Keras model using class parameters
        :return: Keras model
        """
        model_input = keras.Input(shape=(self.num_classes, self.nun_models))

        layer_out = Conv1D(64, kernel_size=self.num_classes, activation="sigmoid")(
            model_input
        )
        layer_out = Conv1D(128, kernel_size=2, activation="linear")(layer_out)
        layer_out = Dropout(0.2)(layer_out)
        layer_out = Conv1D(64, kernel_size=2, activation="linear")(layer_out)
        layer_out = Dense(128)(layer_out)
        layer_out = Dropout(0.2)(layer_out)

        layer_out = Flatten()(layer_out)

        layer_out = Dense(128)(layer_out)
        layer_out = Dropout(0.2)(layer_out)
        output = Dense(self.num_classes, activation="softmax")(layer_out)

        return keras.Model(inputs=model_input, outputs=output)

    def train(self, input_scores: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Trains CNN model to generate softmax scores from
        :param input_scores: numpy array of prediction matrices
        :param true_labels: numpy array of labels corresponding to each prediction matrix
        :return: None
        """
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["acc"]
        )

        input_scores = np.moveaxis(input_scores, 0, -1)

        train_data, val_data, train_labels, val_labels = train_test_split(
            input_scores, true_labels, test_size=0.2, stratify=true_labels
        )

        train_labels = keras.utils.to_categorical(
            train_labels, num_classes=self.num_classes
        )
        val_labels = keras.utils.to_categorical(
            val_labels, num_classes=self.num_classes
        )

        self.model.fit(
            x=train_data,
            y=train_labels,
            validation_data=(val_data, val_labels),
            epochs=30,
            callbacks=[
                TerminateOnNaN(),
                EarlyStopping(
                    patience=15, monitor="val_acc", restore_best_weights=True
                ),
            ],
        )

    def predict(self, prediction_matrix: np.ndarray) -> np.ndarray:
        """
        Generates final predictions using given prediction matrix and trained model
        :param prediction_matrix: numpy array of predictions from ensemble
        :return: numpy array of predictions
        """
        prediction_matrix = np.moveaxis(prediction_matrix, 0, -1)
        softmax_scores = self.model.predict(prediction_matrix)

        return np.argmax(softmax_scores, axis=-1)
