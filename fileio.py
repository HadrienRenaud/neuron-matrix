"""Python module for neuron-matrix.

It implements the main input/output functions used in neuron-matrix.

Author : Hadrien Renaud-Lebret
Created on 2/02/2016
"""


# ********************************* Imports ***********************************
# Imports :

import os

import matplotlib.pyplot as plt
import numpy as np


# ******************************** Functions **********************************
# Functions :


def is_convertible_to_float(string):
    """Function that determine if a string can be safely convert to a float."""
    try:
        float(string)
        return True
    except TypeError:
        return False


def read_sample(file_text):
    """Function reading an sample in a file and returning the corresponding matrix.

    Return the result as a numpy array.
    """
    lines = []
    with open(file_text) as sample_file:
        for line in sample_file:
            if len(line.split()) != 0:
                lines.append([float(elt)
                              for elt in line.split() if is_convertible_to_float(elt)])
    return np.array(lines)


def save_image(matrix, file_name):
    """Function saving matrix as an image."""
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig('test')


def find_examples(directory, alphabet):
    """Iterate through the directory to find all processable examples and return them."""
    assert isinstance(alphabet, str), "Wrong type for alphabet."
    examples = []
    results = []
    try:
        files = os.listdir(directory)
    except OSError as e:
        print("Reading of directory {} failed : {}".format(directory, e))
        return [], []
    for f in files:
        try:
            if os.path.isfile(os.path.join(directory, f)):
                example = read_sample(os.path.join(directory, f))
                result = [int(letter == f[0]) for letter in alphabet]
                examples.append(example)
                results.append(result)
        except OSError as e:
            print("Reading of file {} failed : {}".format(f, e))
    print("find_examples :", len(examples), "files have been found.")
    return examples, results


def learn_on_folder(neurnet, directory, alphabet, learning_algo="default", **args):
    """Make neurnet learn on every example in the directory."""
    learning_algo_possibilities = {
        'default': neurnet.learn,
        'learn': neurnet.learn,
        'learn2': neurnet.learn2,
        '2': neurnet.learn2
    }
    learning_algo_function = learning_algo_possibilities[learning_algo]
    examples, results = find_examples(directory, alphabet)
    for i, ex in enumerate(examples):
        examples[i] = ex.reshape((1, ex.size))[0, :]
    return learning_algo_function(examples, results, **args)

#
# def test_on_folder(neurnet, directory, alphabet, **args):
#     """Make neurnet test every example in the directory."""
#     examples, results = find_examples(directory, alphabet)
#     for i, ex in enumerate(examples):
#         examples[i] = ex.reshape((1, ex.size))[0, :]
#
# ***************************** Excecutable code ******************************
# Excecutable code :
if __name__ == '__main__':
    pass
