"""Python main script file for NeuronProject Matrix flavor.

Author : Hadrien Renaud-Lebret
Created on 28/01/2016
"""


# ********************************* Imports ***********************************
# Imports :

import os
import numpy as np
import matplotlib.pyplot as plt
from neuralnet import NeuralNetwork

# *********************************** Data ************************************
# Data :

learning_sample_folder = 'LearningSample'

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,?;.:!éàè'()+-\"="

length_alphabet = 8

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
                lines.append([float(elt) for elt in line.split() if is_convertible_to_float(elt)])
    return np.array(lines)


def save_image(matrix, file_name):
    """Function saving matrix as an image."""
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig('test')


def find_examples(directory):
    """Iterate through the directory to find all processable examples and return them."""
    examples = []
    results = []
    try:
        files = os.listdir(directory)
    except:
        print("Wrong directory. Aborted.")
        return 0
    for f in files:
        try:
            if os.path.isfile(os.path.join(directory, f)):
                examples.append(read_sample(os.path.join(directory, f)))
                results.append([int(letter == f[0]) for letter in alphabet[:length_alphabet]])
        except:
            print("Reading of {} failed.".format(f))
    print("find_examples :", len(examples), "files have been found.")
    return examples, results


def learn_on_folder(neurnet, directory=learning_sample_folder, **args):
    """Make neurnet learn on every example in the directory."""
    examples, results = find_examples(directory)
    for i, ex in enumerate(examples):
        examples[i] = ex.reshape((1, ex.size))[0, :]
    return neurnet.learn(examples, results, **args)


# ***************************** Excecutable code ******************************
# Excecutable code :
if __name__ == '__main__':
    neur = NeuralNetwork("400:150:50:" + str(length_alphabet))
    neur.randomize_factors()
    print(learn_on_folder(neur, limit_repet=250))
