import numpy as np

from operations.composition import phaser
from operations.helpers import normalize, is_density_matrix
from operations.similarity_measures import k_hyp, trace_similarity, k_e, k_ba
from operations.worldly_context_creation import get_worldly_context, get_contexts


def framework_comparison(density, df, i):
    """
    Generate dataset to compare different negation frameworks. The different frameworks are implemented in the code
        (each saved in a separate column):

    :param density: The density matrices required for the calculation.
    :param df:  The dataframe containing the experiment data and to which the results will be saved.
    :param i:   The row which should be calculated.
    """

    word1_string = df.at[i, 'NEGATED']
    word2_string = df.at[i, 'ALTERNATIVE']

    word1 = density[word1_string]
    word2 = density[word2_string]

    identity = np.identity(word1.shape[0])

    negated = identity - word1
    negated_scaled = identity - 0.8 * word1

    # negated_hyp = identity - k_hyp(word1, identity) * word1
    # negated_hyp_scaled = identity - 0.8 * k_hyp(word1, identity) * word1

    for k in range(4, 5):
        function = lambda x, word1, word2 : (x**(k/2)) * k_hyp(word2, word1).real
        #
        worldly_context = get_worldly_context(word1_string, density, function)
        #
        # combined_negated = phaser(negated, worldly_context, basis='left')
        # combined_negated_scaled = phaser(negated_scaled, worldly_context, basis='left')

        # combined_negated_hyp = phaser(negated_hyp, worldly_context, basis='left')
        # combined_negated_hyp_scaled = phaser(negated_hyp_scaled, worldly_context, basis='left')

        # hyp = k_hyp(word1, worldly_context, check_support=True)

        # combined_wc_minux_hyp = worldly_context - hyp * word1
        # combined_wc_minux_hyp_scaled = worldly_context - 0.8 * hyp * word1

        # context = get_contexts(word1_string, density)
        #
        # combined_context_minus_hyp = np.zeros([50, 50])
        # combined_context_minus_hyp_scaled = np.zeros([50, 50])
        #
        # sum = 0
        # for c in context:
        #     hyp = k_hyp(word1, c[0], check_support=True)
        #
        #     scalar = function(c[1], word1, c[0])
        #     combined_context_minus_hyp += scalar * (c[0] - hyp * word1)
        #     combined_context_minus_hyp_scaled += scalar * (c[0] - 0.8 * hyp * word1)
        #     # print(hyp, 0.8 * hyp)
        #
        #     sum += scalar
        #
        # combined_context_minus_hyp /= sum
        # combined_context_minus_hyp_scaled /= sum

        negations = (
            ("baseline", word1), ("worldly_context", worldly_context),
            # ("logical_negation", negated),
            # ("negated", combined_negated), ("negated_scaled", combined_negated_scaled),
            # ("negated_hyp", combined_negated_hyp), ("negated_hyp_scaled", combined_negated_hyp_scaled),
            # ("wc_minus_hyp", combined_wc_minux_hyp), ("wc_minus_hyp_scaled", combined_wc_minux_hyp_scaled),
            # ("context_minus_hyp", combined_context_minus_hyp),
            # ("context_minus_hyp_scaled", combined_context_minus_hyp_scaled)
        )

        for n in negations:
            normalized = n[1]
            # normalized = normalize(n[1], if_max_eig_always_1=True)

            # df.at[i, str(k) + "_" + n[0] + "_trace"] = trace_similarity(word2, normalized).real
            # df.at[i, str(k) + "_" + n[0] + "_kE1"] = k_e(normalized, word2).real
            # df.at[i, str(k) + "_" + n[0] + "_kE2"] = k_e(word2, normalized).real
            # df.at[i, str(k) + "_" + n[0] + "_khyp1"] = k_hyp(normalized, word2).real
            # df.at[i, str(k) + "_" + n[0] + "_khyp2"] = k_hyp(word2, normalized).real
            df.at[i, str(k) + "_" + n[0] + "_kba"] = k_ba(word2, normalized).real
