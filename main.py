import numpy as np
import pandas as pd
import pickle
import csv

from operations.composition import phaser, fuzz, spider_con
from operations.helpers import FLOAT_PRECISION
from operations.logical_negation import id_neg, supp_ker_inverse
from operations.similarity_measures import trace_similarity, k_hyp
from operations.worldly_context_creation import get_context

density = pickle.load(open("data/density_matrices/density_matrices.p", "rb"))

input_file = "data/alternative_datasets/it_ratings.txt"
output_file = "data/output/test_k_hyp_old.csv"

generation_function = "simple_comparison"


def run():
    file_data = []

    with open(input_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            file_data.append(row)

    df = pd.DataFrame(file_data[1:])
    df.columns = file_data[0]
    print(df.head())
    df["MEANRATING"] = pd.to_numeric(df["MEANRATING"])

    for i in range(0, df['NEGATED'].size):
        run_line(df, i)

    df.to_csv(output_file)


def run_line(df, i):
    if i % 5 == 0:
        print(i)

        if i % 10 == 0 and i != 0:
            df.to_csv(output_file)

    if df.at[i, 'NEGATED'] not in density.keys() or df.at[i, 'ALTERNATIVE'] not in density.keys():
        print("word not found")
        return

    if np.all(abs(density[df.at[i, 'NEGATED']]) < FLOAT_PRECISION):
        print(df.at[i, 'NEGATED'])
        return

    if generation_function == 'simple_comparison':
        simple_comparison(df, i)
    # elif generation_function == 'exponents':
    #     exponents(df, i)
    elif generation_function == 'combine_operations':
        combine_operations(df, i)

    return 1


def simple_comparison(df, i):
    word1_string = df.at[i, 'NEGATED']
    word2_string = df.at[i, 'ALTERNATIVE']

    word1 = density[word1_string]
    word2 = density[word2_string]

    negated = id_neg(word1)

    context = get_context(word1_string, density, lambda x: x ** 3)

    combined = phaser(negated, context, basis='left')

    # df.at[i, 'poly_' + str(j)] = k_E(combined, word2).real
    df.at[i, 'baseline'] = trace_similarity(word2, word1).real
    df.at[i, 'logical'] = trace_similarity(word2, negated).real
    df.at[i, 'conversational'] = trace_similarity(word2, combined).real


# def exponents(df, i):
#     word1_string = df.at[i, 'NEGATED']
#     word2_string = df.at[i, 'ALTERNATIVE']
#
#     word1 = density[word1_string]
#     word2 = density[word2_string]
#
#     negated = id_neg(word1)
#
#     # ------------ exponent calculations
#     for loop_i in range(0, 21):
#         j = loop_i / 2
#         # if not df[i, "poly_" + str(j)].isnull()[i]:
#         #     continue
#
#         function = lambda x: x ** j
#         context = get_context(word1_string, density, function)
#
#         combined = phaser(negated, context, basis='left')
#
#         # df.at[i, 'poly_' + str(j)] = k_E(combined, word2).real
#         df.at[i, 'swap_poly_' + str(j)] = trace_similarity(word2, combined).real
#
#         function = lambda x: (1 + j / 10) ** x
#         context = get_context(word1_string, density, function)
#
#         combined = phaser(negated, context, basis='left')
#
#         # df.at[i, 'exp_' + str(j)] = k_E(combined, word2).real
#         df.at[i, 'swap_exp_' + str(j)] = trace_similarity(word2, combined).real
#
#         function = lambda x: x ** (j / 2)
#
#         left, right, mult = get_context2(word1_string, density, k_E, function)
#
#         combined = phaser(negated, left, basis='left')
#         # df.at[i, 'kE_' + str(j)] = k_E(combined, word2).real
#         df.at[i, 'swap_trace_left_' + str(j)] = trace_similarity(word2, combined).real
#
#         combined = phaser(negated, right, basis='left')
#         # df.at[i, 'kE1_' + str(j)] = k_E(combined, word2).real
#         df.at[i, 'swap_trace_right_' + str(j)] = trace_similarity(word2, combined).real
#
#         combined = phaser(negated, left + right, basis='left')
#         df.at[i, 'swap_trace_both_' + str(j)] = trace_similarity(word2, combined).real
#

hyponomies = [
    ("k_hyp", k_hyp, True),
    # ("k_E", k_E, True),
    # ("k_BA", k_BA, False),
    # ("trace", lambda x, y: np.trace((x / np.trace(x)) @ (y / np.trace(y))), False)
]

context_functions = [
    ("best", lambda x: x ** 3)
]

conjunction_bases = [
    'left',
    'right'
]

conjunctions = [
    # ('mult', mult_con),
    # ('diag', diag_con),
    ('phaser', phaser),
    ('fuzz', fuzz),
    ('spider', spider_con)
]

negations = [
    ("id_neg", lambda x: id_neg(x)),
    ("supp_ker", lambda x: supp_ker_inverse(x))
]


def combine_operations(df, i):
    word1_string = df.at[i, 'NEGATED']
    word2_string = df.at[i, 'ALTERNATIVE']

    word1 = density[word1_string]
    word2 = density[word2_string]

    # ----------------------------- normal loop function
    for context_function in context_functions:
        word1_context = get_context(word1_string, density, context_function[1])

        for negation in negations:
            negation_word1 = (negation[1])(word1)

            for conjunction in conjunctions:
                for base in conjunction_bases:
                    negation_with_cont = conjunction[1](negation_word1, word1_context, basis=base)
                    for hyp in hyponomies:
                        swaps = [("", negation_with_cont, word2)]
                        if hyp[2]:
                            swaps.append(("swap", word2, negation_with_cont))

                        for swap, A, B in swaps:
                            prefix = negation[0] + "_" + conjunction[0] + "_" + base + "_" + swap

                            df.at[i, prefix + hyp[0]] = hyp[1](A, B).real


if __name__ == '__main__':
    run()
