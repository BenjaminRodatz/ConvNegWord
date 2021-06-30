"""
Constants used throughout the code base.

Author: Benjamin Rodatz, Razin Shaikh, Lia Yeh
"""

import numpy as np
import numpy.linalg as la

# precision to be used for floats when checking equivalence
FLOAT_PRECISION = 1e-5


def normalize(rho, if_max_eig_always_1=False):
    """
    Normalization of density matrix to max eigenvalue to
    - less than 1 if if_max_eig_always_1=False
    - exactly 1 otherwise

    :param rho: The density matrix to be normalized.
    :param if_max_eig_always_1: Boolean whether the max eigenvalue should be exactly 1
                alternatively it will be normalized to max 1 (default: False).
    :return: Normalized density matrix.
    """
    if np.all(np.abs(rho) < FLOAT_PRECISION) or \
            (np.all(np.abs(rho) < 1 + FLOAT_PRECISION) and not if_max_eig_always_1):
        return rho

    return rho / np.max(la.eigvalsh(rho))

def get_good_synsets(synset_list):
    good_synsets = []

    sense2freq = []
    for synset in synset_list:
        sense2freq.append(synset.lemmas()[0].count())
    sense2freq = np.array(sense2freq)

    if np.all(sense2freq == 0):
        good_synsets = synset_list
    else:
        sorted_idx = np.argsort(sense2freq)[::-1]

        good_synsets.append(synset_list[sorted_idx[0]])

        if len(sorted_idx) == 2:
            if sense2freq[sorted_idx[1]] != 0:
                good_synsets.append(synset_list[sorted_idx[0]])
        elif len(sorted_idx) > 2 and sense2freq[sorted_idx[1]] != 0:
            same_adj = sense2freq[sorted_idx[1:-1]] == sense2freq[sorted_idx[2:]]
            if False not in same_adj:
                good_synsets = synset_list
            else:
                for i in sorted_idx[1:list(same_adj).index(False) + 2]:
                    good_synsets.append(synset_list[i])
    return good_synsets
