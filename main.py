"""Python main script file for NeuronProject Matrix flavor.

Author : Hadrien Renaud-Lebret
Created on 28/01/2016
"""


# ********************************* Imports ***********************************
# Imports :

import argparse

import fileio as fio
# import getdata as gd
from neuralnet import NeuralNetwork, alphabet, default_values


# ******************************************** Argparse ******************


def argparsor():
    """Return the parser for the programm."""
    parser = argparse.ArgumentParser(description="Show a deep learning example.")

    # general parser
    parser.add_argument("--learning_directory", default=default_values['learning_sample_folder'],
                        help="Directory in which the samples are given")
    parser.add_argument("--length_alphabet", type=int,
                        help="Shortcut for alphabet.", default=default_values['length_alphabet'])
    parser.add_argument("--alphabet", help="Alphabet considered.")
    parser.add_argument('-m', "--momentum", default=default_values['momentum'], type=float)
    parser.add_argument('-l', "--learning_factor",
                        default=default_values['learning_factor'], type=float)

    sub_parsers = parser.add_subparsers(dest='commands')
    parser.set_defaults(func=create_arg_default_function(parser))

    # parser learn
    parser_learn = sub_parsers.add_parser("learn", "l")
    parser_learn.set_defaults(func=arg_learn)
    parser_learn.add_argument('-a', "--learning_algorithm", help='learning_algorithm',
                              choices=['default', 'learn', 'learn2', '2'],
                              type=str, default='default')
    parser_learn.add_argument('-d', "--maximal_distance",
                              default=default_values['maximal_distance'], type=float)
    parser_learn.add_argument('-I', "--limit_iterations",
                              default=default_values['limit_iterations'], type=int)

    # parser learn test
    # parser_learntest = sub_parsers.add_parser("learntest", "lt")
    # parser_learntest.set_defaults(func=arg_learn_test)
    # parser_learntest.add_argument('-a', "--learning_algorithm", help='learning_algorithm',
    #                               choices=['default', 'learn', 'learn2', '2'],
    #                               type=str, default='default')
    # parser_learntest.add_argument('-d', "--maximal_distance",
    #                               default=default_values['maximal_distance'], type=float)
    # parser_learntest.add_argument('-I', "--limit_iterations",
    #                               default=default_values['limit_iterations'], type=int)

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
    print(args, args.length_alphabet)

    print("alphabet", alphabet[:args.length_alphabet], alph)

    # creation of a neural network
    neur = NeuralNetwork("400:150:50:" + str(len(alph)),
                         learning_factor=args.learning_factor,
                         momentum=args.momentum,)
    neur.randomize_factors()

    print(fio.learn_on_folder(neur, args.learning_directory, alph,
                              learning_algo=args.learning_algorithm,
                              limit_repet=args.limit_iterations,
                              max_distance=args.maximal_distance))


# def arg_learn_test(args):
#    """Action to execute on argument learntest."""
#    # alphabet treatment
#    if args.alphabet:
#        alph = args.alphabet
#    else:
#        alph = alphabet[:args.length_alphabet]
#    print(args, args.length_alphabet)
#
#    print("alphabet", alphabet[:args.length_alphabet], alph)
#
#    # creation of a neural network
#    neur = NeuralNetwork("400:150:50:" + str(len(alph)),
#                         learning_factor=args.learning_factor,
#                         momentum=args.momentum,)
#    neur.randomize_factors()
#
#    print(fio.learn_on_folder(neur, args.learning_directory, alph,
#                              learning_algo=args.learning_algorithm,
#                              limit_repet=args.limit_iterations,
#                              max_distance=args.maximal_distance))
#

# **************************************** Excecutable code **************
# Excecutable code :

if __name__ == '__main__':
    args = argparsor().parse_args()
    args.func(args)
