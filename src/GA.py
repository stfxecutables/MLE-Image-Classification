import logging

import numpy as np
import tensorflow as tf


class GA:
    """
    Class to handle genetic algorithm weight matrix optimization
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        population_size: int = 50,
    ) -> None:
        """
        Class initializer.
        :param predictions: numpy array containing prediction matrices (predictions from each base-learner) for all
        label elements
        :param true_labels: class labels for each prediction matrix in `predictions`
        :param population_size: population size for genetic algorithm
        """
        self.predictions = tf.convert_to_tensor(predictions, dtype=tf.float16)
        self.true_labels = tf.convert_to_tensor(true_labels, dtype=tf.int8)
        self.num_models = tf.shape(self.predictions)[0]
        self.population = population_size
        self.num_classes = (tf.shape(self.predictions)[-1]).numpy()
        self.best_weights = None

        def initialize_weights() -> tf.Tensor:
            """
            Initializes weights for genetic algorithm
            :return: Tensor representing weight matrix
            """
            list_of_weights = [
                tf.reshape(
                    tf.repeat(1, self.num_models * self.num_classes),
                    (self.num_models, -1),
                )
            ]
            for _ in range(population_size - 1):
                list_of_weights.append(
                    tf.convert_to_tensor(
                        np.random.randint(
                            -1, 4, self.num_models * self.num_classes
                        ).reshape(self.num_models, -1),
                        dtype=tf.float16,
                    )
                )

            return tf.convert_to_tensor(np.array(list_of_weights), dtype=tf.float16)

        self.weights = initialize_weights()
        self.logger.info(
            f"Weight matrices initialized for population = {population_size}"
        )
        self.generation = 0
        self.fitness_values = self.fitness_all()
        self.best_weights = self.weights[tf.argmax(self.fitness_values)]

    def fitness(self, weights: tf.Tensor) -> int:
        """
        Calculates the fitness (number of correct predictions) for a given weight matrix
        :param weights: Tensor containing one weight matrix
        :return: integer representing fitness (number of correct predictions)
        """
        weighted_predictions = self.predictions * tf.expand_dims(
            tf.expand_dims(weights, axis=-1), axis=-1
        )
        predictions = tf.cast(
            tf.argmax(tf.reduce_sum(weighted_predictions, axis=0), axis=-1), tf.uint8
        )
        incorrect = tf.reduce_sum(
            tf.cast(tf.not_equal(predictions, self.true_labels), tf.int32)
        )

        return tf.shape(self.true_labels)[0] - incorrect

    def fitness_all(self) -> tf.Tensor:
        """
        Calculates the fitness (number of correct predictions) for all individuals (weight matrices) in GA
        :return: List/tensor of fitness values for all individuals in GA
        """
        predictions = tf.expand_dims(self.predictions, 0) * tf.expand_dims(
            self.weights, -2
        )
        predictions = tf.argmax(tf.reduce_sum(predictions, axis=1), axis=-1)
        incorrect = tf.cast(
            tf.not_equal(predictions, tf.cast(self.true_labels, tf.int64)), tf.int32
        )
        incorrect = tf.reduce_sum(incorrect, axis=-1)

        return tf.shape(self.true_labels)[0] - incorrect

    def select(self) -> tf.Tensor:
        """
        Performs selection for crossover for next GA generation
        :return: Tensor containing selected population
        """
        c_prob = tf.cumsum(self.fitness_values / tf.reduce_sum(self.fitness_values))

        new_pop = []

        for i in range(self.population):
            index = tf.where(c_prob > tf.random.uniform(shape=[], dtype=tf.float64))[0][
                0
            ]
            new_pop.append(self.weights[index])

        return tf.convert_to_tensor(np.array(new_pop))

    def crossover(self) -> tf.Tensor:
        """
        Performs crossover for next GA generation
        :return: Tensor containing new population
        """
        # Elitism
        crossover_pop = [self.best_weights]

        for parent1, parent2 in zip(
            range(0, self.population, 2), range(1, self.population, 2)
        ):
            parent1_genes = np.random.randint(
                0, 2, self.num_models * self.num_classes
            ).reshape(self.num_models, -1)
            parent2_genes = np.abs(1 - parent1_genes)
            child_1 = (
                parent1_genes * self.weights[parent1]
                + parent2_genes * self.weights[parent2]
            )
            child_2 = (
                parent2_genes * self.weights[parent1]
                + parent1_genes * self.weights[parent1]
            )

            crossover_pop.append(child_1)
            crossover_pop.append(child_2)

        return tf.convert_to_tensor(np.array(crossover_pop))

    def mutate(self) -> tf.Tensor:
        """
        Performs mutation on current GA population
        :return: Tensor containing mutated population
        """
        index_to_swap = np.random.randint(0, self.population)
        mutated = self.weights.numpy()
        mutated[index_to_swap, :] = np.random.randint(
            -2, 4, self.num_models * self.num_classes
        ).reshape(self.num_models, -1)
        self.logger.info(
            f"Individual {index_to_swap} mutated in generation {self.generation}"
        )

        return tf.convert_to_tensor(mutated, dtype=tf.float16)

    def predict(self, prediction_matrix: np.ndarray) -> np.ndarray:
        """
        Generates final predictions using given prediction matrix and best weight matrix from GA population
        :param prediction_matrix: numpy array of predictions from ensemble
        :return: numpy array of weighted predictions
        """
        weighted_predictions = tf.expand_dims(
            self.best_weights, 1
        ) * tf.convert_to_tensor(prediction_matrix, dtype=tf.float16)
        weighted_predictions = tf.argmax(tf.reduce_sum(weighted_predictions, 0), -1)

        return weighted_predictions.numpy()

    def get_best_accuracy(self) -> float:
        """
        Calculates accuracy of best member in GA population
        :return: float containing accuracy of best member in GA population (accuracy is in range [0.0-1.0]
        """
        return (
            tf.reduce_max(self.fitness_values) / tf.shape(self.true_labels)[0]
        ).numpy()

    def next_gen(self) -> None:
        """
        Performs selection, crossover, mutation for one generation in GA
        :return: None
        """
        self.weights = self.select()
        self.weights = self.crossover()
        if np.random.rand(1) > 0.95:
            self.weights = self.mutate()
        self.fitness_values = self.fitness_all()
        self.best_weights = self.weights[tf.argmax(self.fitness_values)]
        self.logger.info(f"Completed generation: {self}")
        self.generation += 1

    def __repr__(self) -> str:
        """
        generates string representation for GA
        :return: string
        """
        m = tf.reduce_max(self.fitness_values)
        return "Best fitness: {}, \taccuracy: {}, \tGeneration: {}".format(
            m.numpy(), (m / tf.shape(self.true_labels)[0]).numpy(), self.generation
        )
