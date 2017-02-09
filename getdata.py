"""Module for neuron-matrix.

Author : Hadrien Renaud-Lebret
Created on 9/02/2016
"""


# ********************************* Imports ***********************************
# Imports :

from time import time

import fileio as fio
from neuralnet import NeuralNetwork, alphabet, default_values


# ******************************** Functions **********************************
# Functions :


def procedure1(alphabet=alphabet,
               learning_factor=default_values['learning_factor'],
               momentum=default_values['momentum'],
               learning_directory=default_values['learning_sample_folder'],
               learning_algo="default",
               limit_iterations=default_values["limit_iterations"],
               maximal_distance=default_values['maximal_distance'],
               testing_directory=default_values['testing_sample_folder']):
    """Function that execute procedure1.

    It does :
     - create a new NeuralNetwork object
     - randomize its transition_matrix
     - learn on a given dataset
     - test on a given dataset
    """
    neur = NeuralNetwork("400:150:50:" + str(len(alphabet)),
                         learning_factor=learning_factor,
                         momentum=momentum,)
    neur.randomize_factors()

    t0 = time()
    fio.learn_on_folder(neur, learning_directory, alphabet,
                        learning_algo=learning_algo,
                        limit_iterations=limit_iterations,
                        maximal_distance=maximal_distance)
    t1 = time()

    av_dist, succes = fio.test_on_folder(neur, testing_directory, alphabet)
    t2 = time()

    return av_dist, succes, t1 - t0, t2 - t1


def get_data(proc="1", **kwargs):
    """GetData function. Process with different parameters the sample."""
    print(procedure1(**kwargs))


# ***************************** Excecutable code ** ****************************
# Excecutable code :
if __name__ == '__main__':
    pass
