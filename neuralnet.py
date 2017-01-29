"""Python script file, for NeuralNetwork class.

Author : Hadrien Renaud-Lebret
Created on 29/01/2016
"""

import numpy as np

# ******************************************* Functions ********************************************


def inv_cosh(x):
    """Return 1 / cosh(x)."""
    return 1 / np.cosh(x)


def iso_fonction(fonction, mu=1, x0=0):
    """Creator of fonctions, linearly translated."""
    def fonction_bis(x):
        return fonction(mu * x + x0)
    return fonction_bis


def deri_iso_fonction(fonction, mu=1, x0=0):
    """Creator of fonctions, linearly translated, for derivative."""
    def fonction_bis(x):
        return mu * fonction(mu * x + x0)
    return fonction_bis


def learning_progress_display(succes, compt, dist):
    """Display the progress of the learning algorithm."""
    print("Progress : succes {:^7} | compt : {:^7} | dist : {:f}".format(succes, compt, dist))

# ************************************** NeuralNetwork Class ***************************************


class NeuralNetwork:
    """NeuralNetwork class."""

    learning_factor = 0.1

    def __init__(self, geometry,
                 function=np.tanh, function_derivate=inv_cosh, logistic_function_param=(1, 0)):
        """Initialisation of the NeuralNetwork.

        The geometry argument is as string describing the format of the NeuralNetwork:
            '456:12:24:3' will create a network with a first layer with 456 neurons, a second with
            12, a third with 24 and the last with 3.
        function and function_derivate have to be "vecotrized".
        """
        self.geometry = list(map(int, geometry.split(':')))
        self.function = iso_fonction(function,
                                     mu=logistic_function_param[0],
                                     x0=logistic_function_param[1])
        self.function_derivate = deri_iso_fonction(function_derivate,
                                                   mu=logistic_function_param[0],
                                                   x0=logistic_function_param[1])

        # Initialisation of transition_matrix and process_archives
        self.process_archives = [np.zeros(self.geometry[0])]
        self.transition_matrix = []
        for i in range(1, len(self.geometry)):
            self.process_archives.append(np.zeros(self.geometry[i]))
            self.transition_matrix.append(np.zeros((self.geometry[i - 1], self.geometry[i])))

    def set_transition_matrix(self, matrixes):
        """Set the transition_matrix to the correct values."""
        for i, mat in enumerate(matrixes):
            self.transition_matrix[i] = np.array(mat)

    def get_geometry(self):
        """Return self.geometry, modified."""
        return ':'.join(self.geometry)

    def randomize_factors(self):
        """Randomize the transition matrix."""
        for i in range(len(self.geometry) - 1):
            self.transition_matrix[i] = np.random.rand(self.geometry[i], self.geometry[i + 1])

    def apply(self, input_values):
        """Apply the NeuralNetwork to the input values.

        input values as an iterable of numeric values between 0 and 1.
        """
        self.process_archives[0] = 2 * np.array(input_values)[np.newaxis] - 1  # isometry to [-1, 1]
        for i in range(len(self.transition_matrix)):
            self.process_archives[i + 1] = self.function(
                np.dot(self.process_archives[i], self.transition_matrix[i]),)

    def __call__(self, input_values):
        """Apply the Neural Network to the input values.

        DOESN'T SAVE the result for a learning after.
        Use apply in this case.
        input values as an iterable of numeric values between 0 and 1.
        """
        values = 2 * np.array(input_values)[np.newaxis] - 1  # isometry to [-1 , 1]
        for mat in self.transition_matrix:
            values = self.function(np.dot(values, mat))
        return 0.5 + 0.5 * values  # isometry to [0, 1]

    def dist(self, expected_output):
        """Calc the distance of the result to the expected_output."""
        expected_output = np.array(expected_output) * 2 - 1
        return np.sqrt(np.sum((expected_output - self.process_archives[-1])**2))

    def retropropagation(self, expected_output):
        """Apply the retropropagation algorithm."""
        # Initialisation of the error.
        errors = [np.zeros(out.shape) for out in self.process_archives]
        errors[-1] = self.function_derivate(self.process_archives[-1]) * \
            (2 * np.array(expected_output) - 1 - self.process_archives[-1])

        # retropropagation of the errors
        for i in range(len(self.transition_matrix) - 1, 0, -1):
            errors[i] = self.function_derivate(self.process_archives[i]) * \
                np.dot(errors[i + 1], self.transition_matrix[i].transpose())

        # update of the transition_matrix
        for i, mat in enumerate(self.transition_matrix):
            mat += self.learning_factor * \
                np.dot(self.process_archives[i].transpose(), errors[i + 1])

    def learn(self, sample, results, limit_repet=50, max_distance=0.25):
        """Learning algorithm on the given examples."""
        print("NeuralNetwork.learn : begining of the learning algorithm.")
        succes = 0
        compt = 0
        i = 0
        dist = float("infinity")
        while succes < len(sample) and compt < limit_repet * len(sample):
            self.apply(sample[i])
            dist = self.dist(results[i])
            while dist > max_distance and compt < limit_repet * len(sample):
                self.retropropagation(results[i])
                compt += 1
                if compt % 100 == 0:
                    learning_progress_display(succes, compt, dist)
                succes = 0
                self.apply(sample[i])
                dist = self.dist(results[i])
            else:
                succes += 1
            i = (i + 1) % len(sample)

        learning_progress_display(succes, compt, dist)
        print("NeuralNetwork.learn : end of the learning algorithm.")
        return dist

    def to_json(self):
        """Return an expression of the NeuralNetwork in json."""
        dico = {}
        dico['geometry'] = ':'.join(self.geometry)
        dico['transition_matrix'] = [mat.tolist() for mat in self.transition_matrix]
        return dico
