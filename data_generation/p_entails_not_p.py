import random

import numpy as np

from operations.composition import phaser
from operations.helpers import normalize, is_density_matrix
from operations.similarity_measures import k_hyp, trace_similarity, k_e, k_ba
from operations.worldly_context_creation import get_worldly_context, get_contexts


def p_entails_not_p(density, df, i):
    """
    Generate dataset to compare the entailment between p and not p for different frameworks. For this only the first
    word of the column is being used.

    :param density: The density matrices required for the calculation.
    :param df:  The dataframe containing the experiment data and to which the results will be saved.
    :param i:   The row which should be calculated.
    """
    word1_string = df.at[i, 'NEGATED']
    word2_string = df.at[i, 'ALTERNATIVE']

    random.seed(0)
    random_word_string = df.at[random.randint(0, df.shape[0] - 1), 'NEGATED']

    word1 = density[word1_string]
    word2 = density[word2_string]
    random_word = density[random_word_string]

    identity = np.identity(50)

    negated = identity - word1
    negated_scaled = identity - 0.8 * word1

    negated_hyp = identity - k_hyp(word1, identity) * word1
    negated_hyp_scaled = identity - 0.8 * k_hyp(word1, identity) * word1

    for k in range(4, 5):
        function = lambda x: x ** k
        worldly_context = get_worldly_context(word1_string, density, function)

        combined_negated = phaser(negated, worldly_context, basis='left')
        combined_negated_scaled = phaser(negated_scaled, worldly_context, basis='left')

        combined_negated_hyp = phaser(negated_hyp, worldly_context, basis='left')
        combined_negated_hyp_scaled = phaser(negated_hyp_scaled, worldly_context, basis='left')

        hyp = k_hyp(word1, worldly_context, check_support=True)
        combined_wc_minus_hyp = worldly_context - hyp * word1

        combined_wc_minus_hyp_scaled = worldly_context - 0.8 * hyp * word1

        context = get_contexts(word1_string, density)

        combined_context_minus_hyp = np.zeros([50, 50])
        combined_context_minus_hyp_scaled = np.zeros([50, 50])

        sum = 0
        for c in context:
            hyp = k_hyp(word1, c[0], check_support=True)

            scalar = function(c[1])
            combined_context_minus_hyp += scalar * (c[0] - hyp * word1)
            combined_context_minus_hyp_scaled += scalar * (c[0] - 0.8 * hyp * word1)

            sum += scalar

        combined_context_minus_hyp /= sum
        combined_context_minus_hyp_scaled /= sum

        negations = (
            ("random-ish", word2), ("random", random_word),
            ("baseline", word1), ("worldly_context", worldly_context),
            ("negated", combined_negated), ("negated_scaled", combined_negated_scaled),
            ("negated_hyp", combined_negated_hyp), ("negated_hyp_scaled", combined_negated_hyp_scaled),
            ("wc_minus_hyp", combined_wc_minus_hyp), ("wc_minus_hyp_scaled", combined_wc_minus_hyp_scaled),
            ("context_minus_hyp", combined_context_minus_hyp),
            ("context_minus_hyp_scaled", combined_context_minus_hyp_scaled)
        )

        for n in negations:
            # normalized = normalize(n[1], if_max_eig_always_1=True)
            normalized = n[1]

            df.at[i, str(k) + "_" + n[0] + "_trace"] = trace_similarity(word1, normalized).real
            df.at[i, str(k) + "_" + n[0] + "_BA"] = k_ba(word1, normalized).real
            df.at[i, str(k) + "_" + n[0] + "_kE1"] = k_e(word1, normalized).real
            df.at[i, str(k) + "_" + n[0] + "_kE2"] = k_e(normalized, word1).real
            df.at[i, str(k) + "_" + n[0] + "_khyp1"] = k_hyp(word1, normalized).real
            df.at[i, str(k) + "_" + n[0] + "_khyp2"] = k_hyp(normalized, word1).real
