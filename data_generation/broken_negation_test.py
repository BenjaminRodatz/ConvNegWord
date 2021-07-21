import numpy as np

from operations.composition import phaser
from operations.similarity_measures import k_hyp, trace_similarity, k_e
from operations.worldly_context_creation import get_worldly_context, get_contexts


def broken_negation_test(density, df, i):
    """
    Generate dataset to show that our negation is not perfect. The dataset compares four different negations
        (each saved in a separate column):

    1: baseline - simple comparison between word1 and word2
    2: worldly_context - comparison between word2 and worldly context
    3: negated - the negation as presented in the SemSpace paper
    4: broken_hamster - the negation as presented in SemSpace but with logical negation of a fixed word (i.e. hamster)
        instead of the input

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

    for k in range(4, 5):
        function = lambda x: x ** k
        worldly_context = get_worldly_context(word1_string, density, function)

        combined_negated = phaser(negated, worldly_context, basis='left')
        combined_negated_goat = phaser(negated_goat, worldly_context, basis='left')

        negations = (
            ("baseline", word1), ("worldly_context", worldly_context),
            ("negated", combined_negated), ("broken_hamster", combined_negated_goat)
        )

        for n in negations:
            df.at[i, str(k) + "_" + n[0] + "_trace"] = trace_similarity(word2, n[1]).real
            df.at[i, str(k) + "_" + n[0] + "_kE1"] = k_e(word2, n[1]).real
            df.at[i, str(k) + "_" + n[0] + "_kE2"] = k_e(n[1], word2).real
            df.at[i, str(k) + "_" + n[0] + "_khyp1"] = k_hyp(word2, n[1]).real
            df.at[i, str(k) + "_" + n[0] + "_khyp2"] = k_hyp(n[1], word2).real