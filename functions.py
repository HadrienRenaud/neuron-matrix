"""Python module for neuron-matrix.

Implements some useful fonctions.

Created by Hadrien Renaud-Lebret on Feb 16 2017.
"""

import numpy as np


# ****************************** Functions ******************************


def inv_cosh(x):
    r"""Return :math:`\frac{1}{\cosh(x)}`."""
    return 1 / np.cosh(x)


def ReLU(x):
    """Return max(x,0)."""
    return np.fmax(x, 0)


def soft_max(x):
    """Compute the SoftMax of the array x."""
    e = np.exp(x)
    return 2 * e / np.sum(e) - 1


def log_soft_max(x):
    r"""Compute LogSoftMax of the array x.

    :param x: array
    :return: :math:`x_k - \log(\sum_i e^{x_i})`
    """
    alpha = np.log(np.sum(np.exp(x)))
    return x - alpha


def deri_log_soft_max(x):
    r"""Compute the derivative of log_soft_max."""
    e = np.exp(x)
    return 1 - x / np.sum(e)


def one(x):
    """Return 1."""
    return 1


def heaviside(x):
    """Return 1 if x>0, 0 otherwise."""
    return (1 + np.sign(x)) / 2


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
