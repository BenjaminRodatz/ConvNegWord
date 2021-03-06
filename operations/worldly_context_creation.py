from nltk.corpus import wordnet as wn
import numpy as np

from operations.helpers import FLOAT_PRECISION, normalize, get_good_synsets


def get_contexts(word, density_matrices, pos='n'):
    """
    Return the worldly contexts of the given word by fetching them from the WordNet hypernyms of the word.
    Each context is returned as a pair of the density matrix with their distance in the tree to the word.

    :param word:    The words for which the worldly contexts should be calculated.
    :param density_matrices:    A dictionary of all the density matrices necessary.
    :param pos:     The function of the word as utilized in WordNet (default: 'n' - noun)
    :return:        The list of density matrices in form (matrix, tree_depth).
    """
    synsets = wn.synsets(word, pos)

    contexts = list()

    good_synsets = get_good_synsets(synsets)
    for synset in good_synsets:
        for path in synset.hypernym_paths():
            for i in range(len(path)):
                entry = path[i]

                if entry.lemma_names()[0] in density_matrices:
                    contexts.append((density_matrices[entry.lemma_names()[0]], i))

    return contexts


def get_worldly_context(word, density_matrices, scalar_function=lambda i, word1, word2: i ** 4, pos='n', do_normalize=True):
    """
    Return the worldly context of the given word by building it from the WordNet hypernyms of the word.
    The worldly context implemented here is originally described in
    "Conversational Negation using Worldly Context in Compositional Distributional Semantics".

    :param word:    The words for which the worldly context should be calculated.
    :param density_matrices:    A dictionary of all the density matrices necessary.
    :param scalar_function:     The function to calculate the weights of the individual hypernyms.
        (default: f(i) = i**4 - the function which scored highest in experiments)
    :param pos:     The function of the word as utilized in WordNet (default: 'n' - noun).
    :param do_normalize:        Whether to normalize the result of the worldly context (default: True).
    :return:        The density matrix of the worldly context for the given word.
    """
    DENSITY_DIMENSION = density_matrices[(list(density_matrices.keys())[0])].shape[0]

    worldly_context = np.zeros((DENSITY_DIMENSION, DENSITY_DIMENSION))

    contexts = get_contexts(word, density_matrices, pos)

    if contexts == None:
        return np.identity(DENSITY_DIMENSION)

    for context in contexts:
        scalar = scalar_function(context[1], density_matrices[word], context[0])

        worldly_context += scalar * context[0]

    if np.all(abs(worldly_context) < FLOAT_PRECISION):
        return np.identity(DENSITY_DIMENSION)

    if not do_normalize:
        return worldly_context

    return normalize(worldly_context)
