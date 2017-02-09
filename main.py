"""Python main script file for NeuronProject Matrix flavor.

Author : Hadrien Renaud-Lebret
Created on 28/01/2016
"""


# ********************************* Imports ***********************************
# Imports :

import argparse

import fileio as fio
from neuralnet import NeuralNetwork

# *********************************** Data ************************************
# Data :

learning_sample_folder = 'LearningSample'

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,?;.:!éàè'()+-\"="

length_alphabet = 8


# ******************************************** Argparse ******************

def argparsor():
    """Return the parser for the programm."""
    parser = argparse.ArgumentParser(description="Show a deep learning example.")

    parser.add_argument("--learning_directory", default=learning_sample_folder,
                        help="Directory in which the samples are given")
    parser.add_argument("--length_alphabet", type=int,
                        help="Shortcut for alphabet.")
    parser.add_argument("--alphabet", help="Alphabet considered.")

    sub_parsers = parser.add_subparsers(dest='commands')
    parser.set_defaults(func=create_arg_default_function(parser))

    parser_learn = sub_parsers.add_parser("learn")
    parser_learn.set_defaults(func=arg_learn)
    parser_learn.add_argument("--learning_algorithm", help='learning_algorithm',
                              choices=['default', 'learn', 'learn2', '2'],
                              type=str, default='default')

    return parser


def create_arg_default_function(parser):
    """Create a default function, print usage."""
    def print_usage_bis(*args, **kwargs):
        parser.print_usage()
    return print_usage_bis


def arg_learn(args):
    """Action to execute on argument learn."""
    neur = NeuralNetwork("400:150:50:" + str(length_alphabet))
    neur.randomize_factors()

    # alphabet treatment
    if alphabet in args:
        alph = args.alphabet
    elif length_alphabet in args:
        alph = alphabet[:args.length_alphabet]
    else:
        alph = alphabet[:length_alphabet]

    print(fio.learn_on_folder(neur, args.learning_directory, alph,
                              learning_algo=args.learning_algorithm, limit_repet=250))


# **************************************** Excecutable code **************
# Excecutable code :

if __name__ == '__main__':
    args = argparsor().parse_args()
    args.func(args)
