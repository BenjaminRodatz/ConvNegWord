import numpy as np

from operations.composition import phaser
from operations.helpers import normalize, is_density_matrix
from operations.similarity_measures import k_hyp, trace_similarity, k_e
from operations.worldly_context_creation import get_worldly_context, get_contexts


def scalar_comparison(density, df, i):
    """
    Generate dataset to compare different negation frameworks along with their optimal scalar.
    The different frameworks are implemented in the code
        (each saved in a separate column):

    :param density: The density matrices required for the calculation.
    :param df:  The dataframe containing the experiment data and to which the results will be saved.
    :param i:   The row which should be calculated.
    """

    for scalar in {1, 0.8, 0.6, 0.4, 0.2, 0}:
        word1_string = df.at[i, 'NEGATED']
        word2_string = df.at[i, 'ALTERNATIVE']

        word1 = density[word1_string]
        word2 = density[word2_string]

        identity = np.identity(word1.shape[0])

        negated = identity - scalar * word1

        negated_hyp = identity - scalar * k_hyp(word1, identity) * word1

        function = lambda x: x ** 4

        worldly_context = get_worldly_context(word1_string, density, function)

        combined_negated = phaser(negated, worldly_context, basis='left')

        combined_negated_hyp = phaser(negated_hyp, worldly_context, basis='left')

        hyp = k_hyp(word1, worldly_context, check_support=True)

        combined_wc_minux_hyp = worldly_context - scalar * hyp * word1

        context = get_contexts(word1_string, density)

        combined_context_minus_hyp = np.zeros([50, 50])

        sum = 0
        for c in context:
            hyp = k_hyp(word1, c[0], check_support=True)

            weight = function(c[1])
            combined_context_minus_hyp += weight * (c[0] - scalar * hyp * word1)
            # print(hyp, 0.8 * hyp)

            sum += weight

        combined_context_minus_hyp /= sum

        negations = (
            ("negated_" + str(scalar), combined_negated),
            ("negated_hyp_" + str(scalar), combined_negated_hyp),
            ("wc_minus_hyp_" + str(scalar), combined_wc_minux_hyp),
            ("context_minus_hyp_" + str(scalar), combined_context_minus_hyp)
        )

        for n in negations:
            # normalized = n[1]
            normalized = normalize(n[1], if_max_eig_always_1=True)

            df.at[i, str(4) + "_" + n[0] + "_trace"] = trace_similarity(word2, normalized).real
            df.at[i, str(4) + "_" + n[0] + "_kE1"] = k_e(word2, normalized).real
            df.at[i, str(4) + "_" + n[0] + "_kE2"] = k_e(normalized, word2).real
            df.at[i, str(4) + "_" + n[0] + "_khyp1"] = k_hyp(word2, normalized).real
            df.at[i, str(4) + "_" + n[0] + "_khyp2"] = k_hyp(normalized, word2).real
