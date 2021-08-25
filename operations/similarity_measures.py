"""
Similarity and entailment measures between two matrices as proposed in various papers.

Author: Benjamin Rodatz, Razin Shaikh, Lia Yeh
"""

import numpy as np
import numpy.linalg as LA

from operations.helpers import FLOAT_PRECISION, is_density_matrix


def k_ba(rho_a, rho_b):
    """
    k_BA entailment measure as proposed in "Modelling hyponymy in DisCoCat" -- Martha Lewis.
    This metric is not symmetric.

    :param rho_a: First matrix.
    :param rho_b: Second matrix.
    :return: The entailment score.
    """
    if np.all(abs(rho_a) < FLOAT_PRECISION) or np.all(abs(rho_b) < FLOAT_PRECISION):
        return 0

    eg_vals = LA.eigvalsh(rho_b - rho_a)

    # if all eigenvalues are 0, then rhoB = rhoA
    if np.all(abs(eg_vals)) < FLOAT_PRECISION:
        return 1

    return complex(np.sum(eg_vals) / np.sum(np.abs(eg_vals))).real


def __get_error_term(rho_a, rho_b):
    """
    Find error term E such that rho_b - rho_a + E is positive as outlined in
    "Modelling hyponymy in DisCoCat" -- Martha Lewis.

    :param rho_a: First matrix.
    :param rho_b: Second matrix.
    :return: Error term matrix.
    """
    egval, egvec = LA.eigh(rho_b - rho_a)

    egval[np.where(egval > 0)[0]] = 0  # set positive eigenvalues to 0
    egval[np.where(egval < 0)[0]] *= -1  # change sign of negative eigenvalues

    return egvec @ np.diagflat(egval) @ LA.inv(egvec)


def k_e(rho_a, rho_b):
    """
    k_e entailment measure as proposed in "Modelling hyponymy in DisCoCat" -- Martha Lewis.
    This metric is not symmetric.

    :param rho_a: First matrix.
    :param rho_b: Second matrix.
    :return: The entailment score.
    """

    if np.all(abs(rho_a) < FLOAT_PRECISION) or np.all(abs(rho_b) < FLOAT_PRECISION):
        return 0

    rho_e = __get_error_term(rho_a, rho_b)

    return complex(1 - (LA.norm(rho_e) / LA.norm(rho_a))).real


def k_hyp(rho_a, rho_b, check_support=False):
    """
    Generalized version of k_hyp entailment measure as proposed in
    "Graded Entailment for Compositional Distributional Semantics".
    Implemented using theorem 2. The generalization comes from lifting the restriction that the support of rho_a has
    to be a subset of the support of rho_b, as originally required by the theorem. This gives substantially better
    experimental results. This metric is not symmetric.

    :param rho_a: First matrix.
    :param rho_b: Second matrix.
    :param check_support: Whether to check if supp(rho_a) \subseteq supp(rho_b).
    :return: The entailment score.
    """

    if np.all(abs(rho_a) < FLOAT_PRECISION) or np.all(abs(rho_b) < FLOAT_PRECISION):
        return 0

    B_plus = LA.pinv(rho_b)
    Bp_A = B_plus @ rho_a
    egval = LA.eigvals(Bp_A)

    if np.any(egval < -FLOAT_PRECISION):  # check for negative eigenvalues
        return 0
    elif np.max(egval) < FLOAT_PRECISION:  # check if all eigenvalues are 0
        return 0

    value = complex(min(1 / np.max(egval), 1)).real

    if check_support:
        if not is_density_matrix(rho_b - (value / 2) * rho_a):
            return 0

    return value


def trace_similarity(rho_a, rho_b):
    """
    Generalized version of k_hyp entailment measure as proposed in
    "Graded Entailment for Compositional Distributional Semantics".
    Implemented using theorem 2. The generalization comes from lifting the restriction that the support of rho_a has
    to be a subset of the support of rho_b, as originally required by the theorem. This gives substantially better
    experimental results. This metric is symmetric.

    :param rho_a: First matrix.
    :param rho_b: Second matrix.
    :return: The trace similarity.
    """
    return np.trace((rho_a / np.trace(rho_a)) @ (rho_b / np.trace(rho_b)))


def forbenius_norm_similarity(rho_a, rho_b):
    """
    Simple similarity measure between matrices based on the Frobenius norm of the difference.

    :param rho_a: First density matrix.
    :param rho_b: Second density matrix.
    :return: Frobenius norm of rho_a - rho_b
    """
    return np.linalg.norm(rho_a - rho_b, 'fro')
