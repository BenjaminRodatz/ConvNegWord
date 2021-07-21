import numpy as np
import numpy.linalg as LA

from operations.helpers import FLOAT_PRECISION, normalize


def id_neg(rho):
    """
    Calculate the identity inverse of the given density matrix. The identity inverse is proposed in
    "Towards logical negation for compositional distributional semantics" as an operation for logical negation.

    :param rho: The density matrix to be inverted.
    :return: The density matrix resulting from the negation.
    """
    n = rho.shape[0]
    return np.identity(n) - rho


def support_inverse(rho):
    """
    Calculate the support inverse of the given density matrix (also known as Moore-Penrose inverse). The operation
    is described in "Conversational Negation using Worldly Context in Compositional Distributional Semantics".
    It inverts the the support of a matrix and leaves the kernel untouched.

    :param rho: The density matrix which is supposed to be inverted.
    :return:    The density matrix resulting from the inversion.
    """
    return LA.pinv(rho)


def kernel_inverse(rho):
    """
    Calculate the kernes inverse of the given density matrix. The operation is defined in
    "Conversational Negation using Worldly Context in Compositional Distributional Semantics".
    It takes the identity over the kernel of a matrix and sets the support to 0.

    :param rho: The density matrix which is supposed to be inverted.
    :return:    The density matrix resulting from the inversion.
    """
    egval, egvec = LA.eigh(rho)

    if np.any(egval < -FLOAT_PRECISION):
        raise ValueError('Negative eigenvalues of given density matrix')

    pos_egval_idx = np.where(egval > FLOAT_PRECISION)[0]
    zero_egval_idx = np.where(egval < FLOAT_PRECISION)[0]

    egval[pos_egval_idx] = 0
    egval[zero_egval_idx] = 1

    rho_inv = egvec @ np.diagflat(egval) @ LA.inv(egvec)

    return rho_inv


def supp_ker_inverse(rho, s=1, k=1, normalize_supp_inv=False):
    """
    Calculate the support kernes inverse of the given density matrix. The operation is defined in
    "Conversational Negation using Worldly Context in Compositional Distributional Semantics".
    It is defined as a linear combination of the support and the kernel inverse.

    :param rho: The density matrix which is supposed to be negated.
    :param s:   Weight of the support inverse (default: 1)
    :param k:   Weight of the kernel inverse (default: 1)
    :param normalize_supp_inv:  Whether the result should be normalized (default: False)
    :return:    The density matrix resulting from the negation.
    """

    # TODO: check if normalize support inverse
    inverse = s*normalize(support_inverse(rho)) + k*kernel_inverse(rho)

    if normalize_supp_inv:
        return normalize(inverse)
    else:
        return inverse
