import numpy as np

from operations.composition import phaser, fuzz, spider_con, diag_con
from operations.helpers import normalize
from operations.logical_negation import id_neg, supp_ker_inverse, support_inverse, kernel_inverse
from operations.similarity_measures import k_hyp, k_e, k_ba, trace_similarity
from operations.worldly_context_creation import get_worldly_context, get_contexts

# Operations to be explored:
hyponomies = [
    ("khyp", k_hyp, True),
    ("kE", k_e, True),
    ("kBA", k_ba, False),
    ("trace", trace_similarity, False)
]

context_functions = [
    ("hypkhyp", lambda i, word1, word2 : (i**2) * k_hyp(word2, word1).real)
]

conjunction_bases = [
    'left',
    'right'
]

conjunctions = [
    # ('mult', mult_con),
    ('diag', diag_con),
    ('phaser', phaser),
    ('fuzz', fuzz),
    ('spider', spider_con)
]

negations = [
    ("support", lambda x: support_inverse(x)),
    ("kernel", lambda x: kernel_inverse(x)),
    ("suppker", lambda x: supp_ker_inverse(x)),
    ("subtract", lambda x: id_neg(x))
]


def framework1_compare_operations(density, df, i):
    """
    Generate dataset to compare different operations in the first framework proposed in the MSc thesis.
    Each is saved in a different column.

    :param density: The density matrices required for the calculation.
    :param df:  The dataframe containing the experiment data and to which the results will be saved.
    :param i:   The row which should be calculated.
    """

    word1_string = df.at[i, 'NEGATED']
    word2_string = df.at[i, 'ALTERNATIVE']

    word1 = density[word1_string]
    word2 = density[word2_string]

    # ----------------------------- normal loop function

    word1_context = get_contexts(word1_string, density)

    for context_function in context_functions:
        wc = get_worldly_context(word1_string, density, context_function[1])
        if (wc == np.identity(50)).all():
            print(wc == np.identity(50))
            print(wc)
            word1_context = [(wc, 1)]
            print("now")

        for negation in negations:
            negation_word1 = (negation[1])(word1)

            for conjunction in conjunctions:
                for base in conjunction_bases:
                    negation_with_cont = np.zeros([50, 50])

                    scalar_sum = 0
                    for c in word1_context:
                        scalar_sum += (context_function[1])(c[1], word1, c[0])

                    for c in word1_context:
                        scalar = (context_function[1])(c[1], word1, c[0])
                        negation_with_cont += (scalar / scalar_sum) * (conjunction[1](negation_word1, c[0], basis=base))

                    negation_with_cont = normalize(negation_with_cont)

                    for hyp in hyponomies:
                        swaps = [("", negation_with_cont, word2)]
                        if hyp[2]:
                            swaps.append(("swap", word2, negation_with_cont))

                        for swap, A, B in swaps:
                            prefix = negation[0] + "_" + conjunction[0] + "_" + base + "_" + swap

                            df.at[i, prefix + hyp[0]] = hyp[1](A, B).real