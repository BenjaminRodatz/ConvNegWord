import numpy as np

from operations.composition import phaser, fuzz, spider_con, diag_con
from operations.helpers import normalize
from operations.logical_negation import id_neg, supp_ker_inverse, support_inverse, kernel_inverse
from operations.similarity_measures import k_hyp, k_e, k_ba, trace_similarity
from operations.worldly_context_creation import get_worldly_context, get_contexts

# uncommentend are the context functions displayed in the thesis.
context_functions = [
    ("poly", lambda k, x, word1, word2 : k**x),
    ("exp", lambda k, x, word1, word2: (1 + x/10)**k),
    ("hypkE1", lambda k, x, word1, word2 : (k**(x/2)) * k_e(word1, word2).real),
    # ("hypkE2", lambda k, x, word1, word2 : (k**(x/2)) * k_e(word2, word1).real),
    # ("hypkhyp1", lambda k, x, word1, word2 : (k**(x/2)) * k_hyp(word1, word2).real),
    ("hypkhyp2", lambda k, x, word1, word2 : (k**(x/2)) * k_hyp(word2, word1).real),
    # ("hypkBA", lambda k, x, word1, word2 : (k**(x/2)) * abs(k_ba(word2, word1)).real),
    # ("hyptrace", lambda k, x, word1, word2 : (k**(x/2)) * trace_similarity(word2, word1).real)
]

hyponomies = [
    # ("kE", k_e),
    ("trace", trace_similarity)
]

def context_comparison(density, df, i):
    """
    Generate dataset to compare different context function for framework1. Each is saved in a different column.

    :param density: The density matrices required for the calculation.
    :param df:  The dataframe containing the experiment data and to which the results will be saved.
    :param i:   The row which should be calculated.
    """

    word1_string = df.at[i, 'NEGATED']
    word2_string = df.at[i, 'ALTERNATIVE']

    word1 = density[word1_string]
    word2 = density[word2_string]

    negation_word1 = id_neg(word1)
    # ----------------------------- normal loop function
    for context_function in context_functions:
        contexts = get_contexts(word1_string, density)
        # print("--------------", context_function[0])

        for k in range(0, 31):
            x = k / 3
            wc = np.zeros([50, 50])
            sum = 0
            for context in contexts:
                scalar = context_function[1](context[1], x, word1, context[0])
                sum += scalar
                # wc += scalar * context[0]

            negation_with_cont = np.zeros([50, 50])

            for c in contexts:
                scalar = context_function[1](c[1], x, word1, c[0])
                negation_with_cont += (scalar / sum) * (phaser(negation_word1, c[0], basis='left'))

            negation_with_cont = normalize(negation_with_cont)

            wc /= sum

            for hyp in hyponomies:
                prefix = context_function[0] + "_" + hyp[0] + "_" + str(x)

                df.at[i, prefix] = hyp[1](word2, negation_with_cont).real
