"""
This script generates density matrices from vectors using WordNet to encode entailment information.

Author: Benjamin Rodatz, Razin Shaikh, Lia Yeh

Many thanks to Martha Lewis, who's code was used as a basis for this matrix generation.
"""
import csv
import pickle

import nltk
import numpy as np

from operations.helpers import FLOAT_PRECISION, get_good_synsets

nltk.download('wordnet', quiet=True)

from nltk.corpus import wordnet as wn


def hypo(s):
    return s.hyponyms()


def hyper(s):
    return s.hypernyms()


def get_all_hypernyms(words):
    """
    Given an iterable objects containing the words for which the density matrices should be created, this function
    returns a set containing the words and all hypernyms of these words
    (which will be required for the calculation of the density matrices)

    :param words: Iterable object of words for which the density matrices should be calculated
    :return: Set of all hypernyms of words (including words themselves)
    """
    all_words = set()

    for word in words:
        for sym in wn.synsets(word):
            for finding in sym.closure(hyper):
                all_words.update(finding.lemma_names())
        all_words.add(word)

    return all_words


def get_word_hyponyms(word, pos='n', depth=10):
    """
    Given a word, gather all hyponyms of all synsets of the word.

    :param word: The word for which the hyponyms should be generated.
    :param pos: The pos of the word (default: 'n' - noun).
    :param depth: The depth to which hyponyms should be gathered (default: 10)
    :return: a list of hyponyms of the given word
    """

    hyponyms = []

    synset_list = wn.synsets(word, pos=pos)
    if len(synset_list) > 0:
        good_synsets = get_good_synsets(synset_list)

        for synset in good_synsets:
            # collect all the synsets below a given synset
            synsets = list(synset.closure(hypo, depth=depth))
            # include the synset itself as well
            synsets.append(synset)

            for s in synsets:
                for ln in s.lemma_names():
                    hyponyms.append(ln.lower())
        hyponyms = list(set(hyponyms))
    else:
        hyponyms = 'OOV'

    return hyponyms


def build_density_matrix(hypos, dim, hypo_vectors):
    """
    Build the density matrix for a given word with the hyponyms specified in hypos by taking all the vectors of the
    hyponyms and summing their outer products.

    :param hypos: The hyponyms of the word.
    :param dim:  The dimension of the vectors.
    :param hypo_vectors: All the (GloVe) vectors.
    :return: The density matrix of the word.
    """
    matrix = np.zeros([dim, dim])
    if hypos == 'OOV':
        return 'OOV'

    for hyp in hypos:
        if hyp not in hypo_vectors:
            continue
        v = hypo_vectors[hyp]
        vv = np.outer(v, v)
        matrix += vv

    v = matrix
    if np.all(abs(v) < FLOAT_PRECISION):
        return 'OOV'

    max_eig = np.max(np.linalg.eigvalsh(v))
    assert not abs(max_eig) < FLOAT_PRECISION, "Max eigenvalue is 0, should be OOV"
    if max_eig > 1:
        v = v / max_eig
        matrix = v

    return matrix


def get_words_from_file(file):
    """
    Given a csv file of the alternatives dataset as created by Kruszewski et. al. extract the words for which
    we want to create density matrices.

    :param file: Name of the dataset.
    :return: A set containing all the words for which we want to create density matrices.
    """

    words = set()
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            for word in {row[0], row[1]}:
                # add the first two words of each row to the set
                words.add(word)

    return words


def get_vectors(hypo_dict, vec_file, normalisation=False):
    """
    Given a hyponym dictionary, fetch all vectors from a (GloVe) vector file that will become relevant.

    :param hypo_dict:  The hyponym dictionary for which the vectors should be fetched.
    :param vec_file:  The file containing all the (GloVe) vectors.
    :param normalisation: Whether the vectors should be normalized to 1 (default: False).
    :return: A dictionary of all the relevant vectors that are present in the vector file.
    """
    vocabulary = set([hyp for word in hypo_dict for hyp in hypo_dict[word] if hypo_dict[word] != 'OOV'])
    hypo_vectors = {}
    with open(vec_file, 'r') as vf:
        for line in vf:
            entry = line.split()
            if entry[0] in vocabulary:
                vec = np.array([float(n) for n in entry[1:]])
                if normalisation:
                    vec = vec / np.linalg.norm(vec)
                hypo_vectors[entry[0]] = vec
    return hypo_vectors


def run_density_matrix_generation(data_set, vector_file,
                                  output_file_name="../data/density_matrices/density_matrices.p"):
    """
    Given a dataset as created by Kruszewski et. al., create the density matrices for the words in the dataset as well
    as the hypernyms required for the worldly context.

    :param data_set: Location of the dataset.
    :param vector_file: The file where the (GloVe) vectors are stored.
    :param output_file_name: Location where the pickle of the density matrices should be stored.
    """

    print("Fetching words from Kruszewski et. al. data")
    words = get_words_from_file(data_set)
    print("The number of unique words in the dataset is: ", len(words))

    print("Generating complete list of words (including the ones required for worldly context)")
    all_words = get_all_hypernyms(words)

    hypo_dict = dict()

    count = 0
    print("Fetching hyponyms")
    for word in all_words:
        count += 1

        hypo_dict[word] = get_word_hyponyms(word)

        if count % 100 == 0 or count == len(all_words):
            print("Got the hyponyms of ", count, " words out of ", len(all_words))

    with open("../data/density_matrices/all-hypos-temp.p", "wb") as outfile:
        pickle.dump(hypo_dict, outfile)

    print("Fetching vectors")
    vectors = get_vectors(hypo_dict, vector_file)

    print("Building density matrices")

    dim = len((list(vectors.values()))[0])
    matrices = dict()

    count = 0
    for word in hypo_dict.keys():
        count += 1

        if hypo_dict[word] != 'OOV':
            matrices[word] = build_density_matrix(hypo_dict[word], dim, vectors)

        if count % 100 == 0 or count == len(hypo_dict.keys()):
            print("Built the density matrix of ", count, " words out of ", len(hypo_dict.keys()))

    print("Saving results")
    with open(output_file_name, "wb") as dm_file:
        pickle.dump(matrices, dm_file)

    print("Done")


run_density_matrix_generation("../data/alternative_datasets/it_ratings.txt", "../data/glove/glove.6B.50d.txt")
