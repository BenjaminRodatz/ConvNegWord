import math

import numpy as np

from operations.composition import phaser
from operations.helpers import normalize
from operations.logical_negation import id_neg
from operations.similarity_measures import k_hyp, trace_similarity, k_e, k_ba, forbenius_norm_similarity
from operations.worldly_context_creation import get_worldly_context, get_contexts


def broken_negation_test(density, df, i):
    """
    Generate dataset to show that our negation is not perfect. The dataset compares four different negations
        (each saved in a separate column):

    1: baseline - simple comparison between word1 and word2
    2: worldlyContext - comparison between word2 and worldly context
    3: frameworki - the negation as presented in the SemSpace paper
    4: frameworkiGoat - the negation as presented in SemSpace but with logical negation of a fixed word (i.e. goat)
        instead of the input

    For i = 1, 3 and 4. (as framework1 and framework2 are identical under this configuration)

    The reason for the framework still giving reasonable results, even when fixing 'goat' is that the impact of the
        input to the logical negation is smaller than expected. This is further explored in the thesis.

    Additionally it compares the frameworks pairwise.

    :param density: The density matrices required for the calculation.
    :param df:  The dataframe containing the experiment data and to which the results will be saved.
    :param i:   The row which should be calculated.
    """
    word1_string = df.at[i, 'NEGATED']
    word2_string = df.at[i, 'ALTERNATIVE']

    word1 = density[word1_string]
    word2 = density[word2_string]

    goat = density['hamster']

    identity = np.identity(50)

    negated = identity - word1

    negated_goat = identity - goat

    # function = lambda i, word1, word2 : (i**2) * k_hyp(word2, word1).real      #x = 4 thus exponent is 4/2 = 2
    function = lambda i, word1, word2 : i ** 4
    worldly_context = get_worldly_context(word1_string, density, function)

    framework1 = phaser(negated, worldly_context, basis='left')
    framework1_goat = phaser(negated_goat, worldly_context, basis='left')

    context = get_contexts(word1_string, density)
    framework3 = np.zeros([50, 50])
    framework3_goat = np.zeros([50, 50])

    sum = 0
    for c in context:
        hyp = k_hyp(word1, c[0], check_support=True)

        scalar = function(c[1], word1, c[0])
        framework3 += scalar * (c[0] - hyp * word1)

        hyp_goat = k_hyp(goat, c[0], check_support=True)
        framework3_goat += scalar * (c[0] - hyp_goat * goat)

        sum += scalar

    framework3 /= sum
    framework3_goat /= sum

    framework4 = worldly_context - k_hyp(word1, worldly_context, check_support=True) * word1
    framework4_goat = worldly_context - k_hyp(goat, worldly_context, check_support=True) * goat

    # print(k_hyp(goat, worldly_context, check_support=True))

    negations = (
        ("baseline", normalize(word1)), ("worldlyContext", normalize(worldly_context)),
        ("framework1", normalize(framework1)), ("framework1Goat", normalize(framework1_goat)),
        ("framework3", normalize(framework3)), ("framework3Goat", normalize(framework3_goat)),
        ("framework4", normalize(framework4)), ("framework4Goat", normalize(framework4_goat)),
        ("logical", id_neg(word1))
    )

    for n in negations:
        df.at[i, n[0] + "_trace"] = trace_similarity(word2, n[1]).real
        df.at[i, n[0] + "_kE1"] = k_e(n[1], word2).real
        df.at[i, n[0] + "_kE2"] = k_e(word2, n[1]).real
        df.at[i, n[0] + "_khyp1"] = k_hyp(n[1], word2).real
        df.at[i, n[0] + "_khyp2"] = k_hyp(word2, n[1]).real
        df.at[i, n[0] + "_kBA"] = k_ba(word2, n[1]).real


    # pair-wise compare output of operations:
    for loop1 in range(0, len(negations)):
        for loop2 in range(loop1, len(negations)):
            df.at[i, negations[loop1][0] + "_" + negations[loop2][0]] = forbenius_norm_similarity(negations[loop1][1], negations[loop2][1])