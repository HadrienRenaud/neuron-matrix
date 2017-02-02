"""Python main script file for NeuronProject Matrix flavor.

Author : Hadrien Renaud-Lebret
Created on 28/01/2016
"""


# ********************************* Imports ***********************************
# Imports :

import argparse

import fileio as fio
from neuralnet import NeuralNetwork


# ******************************************** Argparse ********************************************

def argparsor():
    """Return the parser for the programm."""
    parser = argparse.ArgumentParser(description="Show a deep learning example.")
    sub_parsers = parser.add_subparsers(dest='commands')
    parser.set_defaults(func=parser.print_usage)

    parser_learn = sub_parsers.add_parser("learn")
    parser_learn.set_defaults(func=arg_learn)

    return parser


def arg_default_function():
    """Default function, do nothing."""
    pass


def arg_learn():
    """Action to execute on argument learn."""
    neur = NeuralNetwork("400:150:50:" + str(fio.length_alphabet))
    neur.randomize_factors()
    print(fio.learn_on_folder(neur, limit_repet=250))


# **************************************** Excecutable code ****************************************
# Excecutable code :

if __name__ == '__main__':
    args = argparsor().parse_args()
    args.func()
