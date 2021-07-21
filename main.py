import numpy as np
import pandas as pd
import pickle
import csv

from data_generation.broken_negation_test import broken_negation_test
from data_generation.compare_operations import *
from data_generation.framework_comparison import framework_comparison
from data_generation.p_entails_not_p import p_entails_not_p
from operations.helpers import FLOAT_PRECISION

density = pickle.load(open("data/density_matrices/density_matrices.p", "rb"))

input_file = "data/alternative_datasets/it_ratings.txt"
output_file = "data/output/test.csv"

generation_function = framework_comparison


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

    df.to_csv(output_file, index=False)


def run_line(df, i):
    if i % 20 == 0:
        print(i, " out of ", df['NEGATED'].size)

        if i % 10 == 0 and i != 0:
            df.to_csv(output_file, index=False)

    if df.at[i, 'NEGATED'] not in density.keys() or df.at[i, 'ALTERNATIVE'] not in density.keys():
        print("word not found")
        return

    if np.all(abs(density[df.at[i, 'NEGATED']]) < FLOAT_PRECISION):
        print(df.at[i, 'NEGATED'])
        return

    generation_function(density, df, i)

    return 1


if __name__ == '__main__':
    run()
