import math

import numpy as np
import numpy.linalg as la

from operations.helpers import FLOAT_PRECISION, normalize


def __and_matrix(n, basis=None):
    """
    Create a helper matrix for the spider composition.

    :param n: The dimension of the matrix.
    :param basis: The basis of the matrix.
    :return: The helper matrix.
    """
    if basis is None:
        basis = np.identity(n)
    mat = np.zeros((n, n * n))
    for i in range(n):
        mat = mat + np.outer(basis[:, i], np.kron(basis[:, i], basis[:, i]))

    return mat


def spider_con(rho_a, rho_b, if_normalize=True, basis='right'):
    """
    Spider conjuction between rho_a and rho_b.
    
    :param rho_a: First density matrix.
    :param rho_b: Second density matrix.
    :param if_normalize: Boolean whether the result should be normalized (default: True)
    :param basis: Basis used for the spider:
                 'left' - rhoA
                 'right' - rhoB
                 'comp' - computational basis
                 or pass the basis matrix directly (default: 'right')
    :return: The density matrix resulting from applying the spider to the two inputs.
    """
    n = rho_a.shape[0]

    if type(basis) == np.ndarray:
        pass
    elif basis == 'comp':
        basis = np.identity(n)
    elif basis == 'left':
        _, basis = la.eigh(rho_a)
    elif basis == 'right':
        _, basis = la.eigh(rho_b)
    else:
        raise TypeError("basis is not an array, 'left', 'right' or 'comp'")

    and_mat = __and_matrix(n, basis)

    rho = and_mat @ np.kron(rho_a, rho_b) @ and_mat.T

    if if_normalize:
        rho = normalize(rho)

    return rho


def __kmult(rho_a, rho_b):
    """
    The actual operation used to implement the fuzz. We borrow the naming used in
    "Towards logical negation for compositional distributional semantics".

    :param rho_a: First density matrix (this one provides the basis).
    :param rho_b: Second density matrix.
    :return:    The resulting density matrix.
    """
    rho_a = np.real(rho_a)
    rho_b = np.real(rho_b)

    vals, vecs = np.linalg.eigh(rho_a)
    vals = np.real(vals)
    vecs = np.array([vec for val, vec in zip(vals, vecs) if np.abs(val) > FLOAT_PRECISION])
    vals = vals[np.abs(vals) > FLOAT_PRECISION]

    vecs = np.real(vecs)

    result = np.zeros(np.shape(rho_a))
    for vec, val in zip(vecs, vals):
        p = np.outer(vec, vec)
        result += val * p.dot(rho_b.dot(p))
    return result


def fuzz(rho_a, rho_b, if_normalize=True, basis='right'):
    """
    Caclulate the fuzz of rho_a and rho_b as defined in "Meaning updating of density matrices".

    :param rho_a: The first density matrix.
    :param rho_b: The second density matrix.
    :param if_normalize: Boolean whether the final result should be normalized (default: True).
    :param basis: The basis which should be taken for the composition
        'left' - rhoA
        'right' - rhoB (default: 'right')
    :return: The density matrix resulting from the composition.
    """
    # if either matrix is 0 return 0
    if np.all(abs(rho_a) < FLOAT_PRECISION):
        return rho_a
    if np.all(abs(rho_b) < FLOAT_PRECISION):
        return rho_b

    if basis == 'left':
        rho = __kmult(rho_a, rho_b)
    elif basis == 'right':
        rho = __kmult(rho_b, rho_a)
    else:
        raise TypeError("basis is not 'left' or 'right'")

    if if_normalize:
        return normalize(rho)
    else:
        return rho


def __matsqrt(rho):
    """
    For a given density matrix rho = sum_i (lambda_i**2) * rho_i with eigenvectors rho_i calculate the density
    matrix sum_i lambda_i * rho_i i.e. with the square root of the eigenvalues.
    
    :param rho: The density matrix of which the square root should be taken.
    :return: The resulting density matrix.
    """
    rho = np.real(rho)
    vals, vecs = np.linalg.eigh(rho)

    vals = np.real(vals)
    vecs = np.array([vec for val, vec in zip(vals, vecs) if np.abs(val) > FLOAT_PRECISION])

    vals = vals[np.abs(vals) > FLOAT_PRECISION]
    sqrtvals = [math.sqrt(v) for v in vals]

    vecs = np.real(vecs)
    sqrt = vecs.T.dot(np.diag(sqrtvals)).dot(vecs)

    return sqrt


def phaser(rho_a, rho_b, if_normalize=True, basis='right'):
    """
    Caclulate the phaser of rho_a and rho_b as defined in "Meaning updating of density matrices".
    
    :param rho_a: The first density matrix.
    :param rho_b: The second density matrix.
    :param if_normalize: Boolean whether the final result should be normalized (default: True).
    :param basis: The basis which should be taken for the composition
        'left' - rhoA
        'right' - rhoB (default: 'right')
    :return: The density matrix resulting from the composition.
    """
    # if either matrix is 0 return 0
    if np.all(abs(rho_a) < FLOAT_PRECISION):
        return rho_a
    if np.all(abs(rho_b) < FLOAT_PRECISION):
        return rho_b

    if basis == 'right':
        rho = __matsqrt(rho_b) @ rho_a @ __matsqrt(rho_b)
    elif basis == 'left':
        rho = __matsqrt(rho_a) @ rho_b @ __matsqrt(rho_a)
    else:
        raise TypeError("basis is not 'left' or 'right'")

    if if_normalize:
        return normalize(rho)
    else:
        return rho
