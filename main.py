"""Python main script file for NeuronProject Matrix flavor.

Author : Hadrien Renaud-Lebret
Created on 28/01/2016
"""


# ********************************* Imports ***********************************
# Imports :

import argparse

import numpy as np

import fileio as fio
import functions as fu
import getdata as gd
from neuralnet import NeuralNetwork, alphabet, default_values


# ******************************************** Argparse ******************


def argparsor():
    """Return the parser for the programm."""
    parser = argparse.ArgumentParser(description="Show a deep learning example.")

    # general parser
    parser.add_argument("--learning_directory", default=default_values['learning_sample_folder'],
                        help="Directory in which the samples for learnging are given.")
    parser.add_argument('--testing_directory', default=default_values['testing_sample_folder'],
                        help="Directory in which the samples for testing are given.")
    parser.add_argument("--length_alphabet", type=int,
                        help="Shortcut for alphabet.", default=default_values['length_alphabet'])
    parser.add_argument("--alphabet", help="Alphabet considered.")
    parser.add_argument('-m', "--momentum", default=default_values['momentum'], type=float)
    parser.add_argument('-l', "--learning_factor",
                        default=default_values['learning_factor'], type=float)
    parser.add_argument('-a', "--learning_algorithm", help='learning_algorithm',
                              choices=['default', 'learn', 'learn2', '2'],
                              type=str, default=default_values["learning_algorithm"])
    parser.add_argument('-d', "--maximal_distance",
                              default=default_values['maximal_distance'], type=float)
    parser.add_argument('-I', "--limit_iterations",
                              default=default_values['limit_iterations'], type=int)

    sub_parsers = parser.add_subparsers(dest='commands')
    parser.set_defaults(func=create_arg_default_function(parser))

    # parser learn
    parser_learn = sub_parsers.add_parser("learn", aliases=["l"])
    parser_learn.set_defaults(func=arg_learn)

    # parser learn test
    parser_learntest = sub_parsers.add_parser("learntest", aliases=["lt"])
    parser_learntest.set_defaults(func=arg_learn_test)

    # parser getdata
    parser_getdata = sub_parsers.add_parser("getdata", aliases=["gd"])
    parser_getdata.set_defaults(func=arg_getdata)

    return parser


def create_arg_default_function(parser):
    """Create a default function, print usage."""
    def print_usage_bis(*args, **kwargs):
        parser.print_usage()
    return print_usage_bis


def arg_learn(args):
    """Action to execute on argument learn."""
    # alphabet treatment
    if args.alphabet:
        alph = args.alphabet
    else:
        alph = alphabet[:args.length_alphabet]

    # creation of a neural network
    neur = NeuralNetwork("400:100:" + str(len(alph)),
                         learning_factor=args.learning_factor,
                         momentum=args.momentum,
                         functions=[(fu.ReLU, fu.heaviside), (fu.log_soft_max, fu.one)])
    neur.randomize_factors()

    print(fio.learn_on_folder(neur, args.learning_directory, alph,
                              learning_algo=args.learning_algorithm,
                              limit_iterations=args.limit_iterations,
                              maximal_distance=args.maximal_distance))


def arg_learn_test(args):
    """Action to execute on argument learntest."""
    # alphabet treatment
    if args.alphabet:
        alph = args.alphabet
    else:
        alph = alphabet[:args.length_alphabet]

    # creation of a neural network
    neur = NeuralNetwork("400:85:" + str(len(alph)),
                         learning_factor=args.learning_factor,
                         momentum=args.momentum,
                         functions=[(np.tanh, fu.inv_cosh),
                                    (fu.log_soft_max, fu.deri_log_soft_max)])
    neur.randomize_factors()

    res = fio.learn_on_folder(neur, args.learning_directory, alph,
                              learning_algo=args.learning_algorithm,
                              limit_iterations=args.limit_iterations,
                              maximal_distance=args.maximal_distance)
    print("Learning result :", res)

    av_dist, succes = fio.test_on_folder(neur, args.testing_directory, alph)
    print("Test result : {:10.4f} distance on average and {:3.2%} of success on the tests.".format(
        av_dist, succes))


def arg_getdata(args):
    """Action called on getdata."""
    if args.alphabet:
        alph = args.alphabet
    else:
        alph = alphabet[:args.length_alphabet]

    print(gd.get_data(alphabet=alph,
                      learning_directory=args.learning_directory,
                      testing_directory=args.testing_directory))


# **************************************** Excecutable code **************
# Excecutable code :

if __name__ == '__main__':
    args = argparsor().parse_args()
    args.func(args)
