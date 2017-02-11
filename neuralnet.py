"""Python module for NeuralNetwork class.

It provides an implementation of a NeuralNetwork with utilitaries. It is NOT bounded to learning
on images or even to learning on samples in different files.
"""

# ******************************* Imports *******************************

import numpy as np

# ******************************** Data ********************************


alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,?;.:!éàè'()+-\"="

default_values = {
    'length_alphabet': 26,
    'learning_factor': 0.1,
    'momentum': 0.5,
    'maximal_distance': 0.2,
    'limit_iterations': 50,
    'learning_sample_folder': 'LearningSample',
    'testing_sample_folder': 'TestSample',
    'learning_algorithm': "default",
}


# ****************************** Functions ******************************


def inv_cosh(x):
    r"""Return :math:`\frac{1}{\cosh(x)}`."""
    return 1 / np.cosh(x)


def iso_fonction(fonction, mu=1, x0=0):
    r"""Creator of fonctions, linearly translated.

    Compute a isometric transform of fonction with the formula :
    :math:`\forall x, \mathrm{iso\_fonction}(f)(x) = f(\mu \times x + x_0)`

    :param float mu: mutliplicative factor :math:`\mu`
    :param float x0: additive term :math:`x_0`
    :param fonction: function taking 1 numeric positionnal argument.
    :return: :math:`F : x \to f(\mu \times x + x_0)`
    """
    def fonction_bis(x):
        return fonction(mu * x + x0)
    return fonction_bis


def deri_iso_fonction(fonction, mu=1, x0=0):
    r"""Creator of fonctions, affinely translated, for derivative.

    Compute a isometric transform of fonction with the formula :
        :math:`\forall x, \mathrm{deri\_iso\_fonction}(f)(x) = \mu \times f(\mu \times x + x_0)`

    Which is equivalent to, if :math:`f` is the derivative of :math:`F`:
        :math:`\forall x, \mathrm{deri\_iso\_fonction}(f, \mu, x_0)(x) = \frac{d}{dx}(\mathrm{iso\_fonction}(F, \mu, x_0))(x) = \mu \times \frac{dF}{dx}(\mu \times x + x_0) = \mu \times f(\mu \times x + x_0)`

    :note: return the derivate of :func:`iso_fonction`
    :param float mu: mutliplicative factor :math:`\mu`
    :param float x0: additive term :math:`x_0`
    :param fonction: function taking 1 numeric positionnal argument.
    :return: :math:`F : x \to \mu \times f(\mu \times x + x_0)`
    """
    def fonction_bis(x):
        return mu * fonction(mu * x + x0)
    fonction_bis.__doc__ = fonction.__doc__
    fonction_bis.__doc__ += """
     Isometricaly translated with mu = {} and x0 = {}
     """.format(mu, x0)
    return fonction_bis


def learning_progress_display(**args):
    """Display the progress of the learning algorithm."""
    formats = {'succes': "{:^7}", 'compt': "{:^7}", 'dist': "{:8.4f}", 'av_dist': "{:8.4f}"}
    print("Progress : ", end="")
    for arg_name, value in sorted(args.items()):
        print(arg_name, formats[arg_name].format(value), end=' | ')
    print()

# ************************************** NeuralNetwork Class *************


class NeuralNetwork:
    """NeuralNetwork class."""

    def __init__(self, geometry, function=np.tanh, function_derivate=inv_cosh,
                 logistic_function_param=(1, 0), learning_factor=0.1, momentum=0):
        """Initialisation of the NeuralNetwork.

        :param str geometry: string describing the format of the NeuralNetwork:
            '456:12:24:3' will create a network with a first layer with 456 neurons, a second with
            12, a third with 24 and the last with 3.
        :param function: vectorized function (see `numpy.vectorize <https://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html#numpy-vectorize>`_)
        :param function_derivate: its (vectorized) function
        :param tuple logistic_function_param: (mu, x0) parameters send to :iso_fonction: and
            :deri_iso_fonction: slope and offset of the logistic function.
        """
        #: geometry of the NeuralNetwork.
        self.geometry = list(map(int, geometry.split(':')))

        #: learning_factor
        self.learning_factor = learning_factor

        #: inertia factor : between 0 and 1
        self.momentum = momentum

        #: transition function of the NeuralNetwork
        self.function = iso_fonction(function,
                                     mu=logistic_function_param[0],
                                     x0=logistic_function_param[1])

        #: the derivative of the transition function, used in backpropagation
        self.function_derivate = deri_iso_fonction(function_derivate,
                                                   mu=logistic_function_param[0],
                                                   x0=logistic_function_param[1])

        #: process archives, used in backpropagation
        self.process_archives = [np.zeros(self.geometry[0])]
        self.transition_matrix = []  #: weight of the neuron transitions
        #: difference of transitions matrix, used for inertia in backpropagation
        self.transition_matrix_diff = []

        # Initialisation of transition_matrix and process_archives and transition_matrix_diff
        for i in range(1, len(self.geometry)):
            self.process_archives.append(np.zeros(self.geometry[i]))
            self.transition_matrix.append(np.zeros((self.geometry[i - 1], self.geometry[i])))
            self.transition_matrix_diff.append(
                np.zeros((self.geometry[i - 1], self.geometry[i])))

    def set_transition_matrix(self, matrixes):
        """Set the transition_matrix to the correct values."""
        for i, mat in enumerate(matrixes):
            self.transition_matrix[i] = np.array(mat)

    def get_geometry(self):
        """Return self.geometry.

        :return: self.geometry modified to render like the one passed as
            an argument of :func:`~NeuralNetwork.__init__`.
        :rtype: str
        """
        return ':'.join(self.geometry)

    def randomize_factors(self):
        """Randomize the transition matrix."""
        for i in range(len(self.geometry) - 1):
            self.transition_matrix[i] = np.random.rand(self.geometry[i], self.geometry[i + 1])

    def apply(self, input_values):
        """Apply the NeuralNetwork to the input values.

        :param input_values: as an iterable of numeric values between 0 and 1.
        :return: an numpy array of values between 0 and 1.
        """
        self.process_archives[0] = 2 * \
            np.array(input_values)[np.newaxis] - 1  # isometry to [-1, 1]
        for i in range(len(self.transition_matrix)):
            self.process_archives[i + 1] = self.function(
                np.dot(self.process_archives[i], self.transition_matrix[i]),)
        return (0.5 + 0.5 * self.process_archives[-1])[0, ]  # isometry to [0, 1]

    def __call__(self, input_values):
        """Apply the Neural Network to the input values.

        :warning: DOESN'T SAVE the result for a learning after.
            Use :func:`~NeuralNetwork.apply` in this case.

        :param input_values: as an iterable of numeric values between 0 and 1.
        :return: an numpy array of values between 0 and 1.
        """
        values = 2 * np.array(input_values)[np.newaxis] - 1  # isometry to [-1 , 1]
        for mat in self.transition_matrix:
            values = self.function(np.dot(values, mat))
        return (0.5 + 0.5 * values)[0, ]  # isometry to [0, 1]

    def dist(self, expected_output):
        r"""Calc the distance of the result to the expected_output.

        It computes the distance between the results found in
        :data:`~NeuralNetwork.process_archives`
        with the formula :
        :math:`\sqrt{\sum_{i} (y_i - x_{-1, i})^2}`

        :param numpy.array expected_output: expected result :math:`(y_i)_i`
        :note: the distance is not an average distance on the two arrays.
        :note: compute the `euclidian norm <https://en.wikipedia.org/wiki/Euclidean_distance>`_ of
            the difference between the two arrays.
        :return float: :math:`\sqrt{\sum_{i} (y_i - x_i)^2}`
        """
        expected_output = np.array(expected_output) * 2 - 1
        return np.sqrt(np.sum((expected_output - self.process_archives[-1])**2))

    def backpropagation(self, expected_output):
        r"""Apply the backpropagation algorithm.

        :note: You have to :func:`~NeuralNetwork.apply` the Network on the sample before.
        :param numpy.array expected_output: expected results

        :Execution:

        * computing of the errors :

            * Initialisation at the bottom of the NeuralNetwork
                :math:`e_{-1} := f'(x_{-1}) \times (y - x_{-1})`
            * backpropagation of the gradient
                :math:`e_{i-1} := f'(x_{i-1}) \times (e_{i+1} \cdot t_i^T)`

        * correction of the transition matrix :

            * computing of the differencial matrix:
                :math:`\Delta t_i := \tau (1 - \mu) (x_i^T \cdot e_{i+1}) + \mu \Delta t_i`
            * correcting the transition matrix:
                :math:`t_i := t_i + \Delta t_i`
        """
        # Initialisation of the error.
        errors = [np.zeros(out.shape) for out in self.process_archives]
        errors[-1] = self.function_derivate(self.process_archives[-1]) * \
            (2 * np.array(expected_output) - 1 - self.process_archives[-1])

        # backpropagation of the errors
        for i in range(len(self.transition_matrix) - 1, 0, -1):
            errors[i] = self.function_derivate(self.process_archives[i]) * \
                np.dot(errors[i + 1], self.transition_matrix[i].transpose())

        # update of the transition_matrix
        for i, mat in enumerate(self.transition_matrix_diff):
            mat = self.learning_factor * (1 - self.momentum) * \
                np.dot(self.process_archives[i].transpose(), errors[i + 1]) + \
                self.momentum * mat
            self.transition_matrix[i] += mat

    def learn(self, sample, results, limit_iterations=50, maximal_distance=0.25):
        """Learning algorithm on the given examples.

        First algorithm.
        """
        print("NeuralNetwork.learn : begining of the learning algorithm.")

        succes = 0
        compt = 0
        i = 0
        dist = float("infinity")

        while succes < len(sample) and compt < limit_iterations * len(sample):
            self.apply(sample[i])
            dist = self.dist(results[i])
            while dist > maximal_distance and compt < limit_iterations * len(sample):
                self.backpropagation(results[i])
                compt += 1
                if compt % 100 == 0:
                    learning_progress_display(succes=succes, compt=compt, dist=dist)
                succes = 0
                self.apply(sample[i])
                dist = self.dist(results[i])
            else:
                succes += 1
            i = (i + 1) % len(sample)

        learning_progress_display(succes=succes, compt=compt, dist=dist)
        print("NeuralNetwork.learn : end of the learning algorithm.")
        return dist

    def learn2(self, sample, results, limit_iterations=50, maximal_distance=0.25):
        """Learning algorithm on the given examples.

        Method given by Hélène Milhem
        `here <https://moodle.insa-toulouse.fr/file.php/457/ReseauNeurones.pdf>`_.
        """
        print("NeuralNetwork.learn2 : begining of the learning algorithm.")
        compt = 0  # number of iterations

        # average distance on the sample processing
        av_dist = 0
        for i in range(len(sample)):
            self.apply(sample[i])
            av_dist += self.dist(results[i])

        indexes = np.array(range(len(sample)))

        while av_dist > maximal_distance and compt < limit_iterations * len(sample):

            # learning
            np.random.shuffle(indexes)
            for i in indexes:
                self.apply(sample[i])
                self.backpropagation(results[i])

            # distance processing
            av_dist = 0
            for i in range(len(sample)):
                self.apply(sample[i])
                av_dist += self.dist(results[i])
            av_dist = av_dist / len(sample)

            # display
            compt += 1
            if compt % (limit_iterations / 100) == 0:
                learning_progress_display(compt=compt * len(sample), av_dist=av_dist)

        learning_progress_display(compt=compt * len(sample), av_dist=av_dist)
        print("NeuralNetwork.learn2 : end of the learning algorithm.")
        return av_dist

    def to_json(self):
        """Return an expression of the NeuralNetwork in json."""
        dico = {}
        dico['geometry'] = ':'.join(self.geometry)
        dico['transition_matrix'] = [mat.tolist() for mat in self.transition_matrix]
        return dico
