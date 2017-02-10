"""Module for neuron-matrix."""

# ********************************* Imports ***********************************
# Imports :

from time import time

import fileio as fio
from neuralnet import NeuralNetwork, alphabet, default_values


# *********************************** Data ***********************************

default_ranges = {
    'learning_factor': [default_values['learning_factor']],
    'momentum': [default_values['momentum']],
    'learning_algo': [default_values['learning_algorithm']],
    'limit_iterations': [default_values['limit_iterations']],
    'maximal_distance': [default_values['maximal_distance']],
}


# ********************************** Classes **********************************

class IteratorMultiple:
    """Iterator through an unknown number of lists."""

    def __init__(self, lengths):
        """Initialisation."""
        self.lengths = lengths
        self.position = [0 for i in self.lengths]
        if self.position:
            self.position[0] = -1

    def __iter__(self):
        """Iterate."""
        return self

    def __next__(self):
        """Next method."""
        for i, l in enumerate(self.lengths):
            self.position[i] += 1
            if self.position[i] >= l:
                self.position[i] = 0
            else:
                break
        else:
            raise StopIteration
        return self.position


# ******************************** Functions **********************************
# Functions :


def procedure1(alphabet=alphabet,
               learning_algo=default_values['learning_algorithm'],
               learning_directory=default_values['learning_sample_folder'],
               testing_directory=default_values['testing_sample_folder'],
               learning_factor=default_values['learning_factor'],
               momentum=default_values['momentum'],
               limit_iterations=default_values["limit_iterations"],
               maximal_distance=default_values['maximal_distance']):
    """Function that execute procedure1.

    It does:
    - create a new NeuralNetwork object
    - randomize its transition_matrix
    - learn on a given dataset
    - test on a given dataset
    """
    # creation
    neur = NeuralNetwork("400:150:50:" + str(len(alphabet)),
                         learning_factor=learning_factor,
                         momentum=momentum,)

    # randomization
    neur.randomize_factors()

    # learning
    t0 = time()
    fio.learn_on_folder(neur, learning_directory, alphabet,
                        learning_algo=learning_algo,
                        limit_iterations=limit_iterations,
                        maximal_distance=maximal_distance)

    # testing
    t1 = time()
    av_dist, succes = fio.test_on_folder(neur, testing_directory, alphabet)
    t2 = time()

    return av_dist, succes, t1 - t0, t2 - t1


def get_data(proc="1", ranges=default_ranges, **kwargs):
    """Process with different parameters the sample."""
    procs = {'1': procedure1}
    proc = procs[proc]

    ranges_li = sorted(ranges.items())
    ranges_cat = [elt[0] for elt in ranges_li]
    results = []

    print("getdata :", ranges_cat)

    for li in IteratorMultiple([len(elt[1]) for elt in ranges_li]):
        values = [elt[1][pos] for elt, pos in zip(ranges_li, li)]
        res = proc(**dict(zip(ranges_cat, values)))
        results.append((li, res))
        print(res)

    print("getdata done.")
    return ranges_cat, results


# ***************************** Excecutable code ** ****************************
# Excecutable code :

if __name__ == '__main__':
    pass
