

# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import numpy as np

from operations.helpers import FLOAT_PRECISION, normalize, get_good_synsets


def get_context(word, density_matrices, scalar_function=lambda i : i**4, pos='n'):
    """
    Return the worldly context of the given word by building it from the WordNet hypernyms of the word.
    The worldly context implemented here is originally described in
    "Conversational Negation using Worldly Context in Compositional Distributional Semantics".

    :param word:    The words for which the worldly context should be calculated.
    :param density_matrices:    A dictionary of all the density matrices necessary.
    :param scalar_function:     The function to calculate the weights of the individual hypernyms.
        (default: f(i) = i**4 - the function which scored highest in experiments)
    :param pos:     The function of the word as utilized in WordNet (default: 'n' - noun)
    :return:        The density matrix of the worldly context for the given word.
    """
    synsets = wn.synsets(word, pos)

    DENSITY_DIMENSION = density_matrices[(list(density_matrices.keys())[0])].shape[0]

    worldly_context = np.zeros((DENSITY_DIMENSION, DENSITY_DIMENSION))

    good_synsets = get_good_synsets(synsets)
    for synset in good_synsets:
        for path in synset.hypernym_paths():
            for i in range(len(path)):
                entry = path[i]

                scalar = scalar_function(i)

                if entry.lemma_names()[0] in density_matrices:
                    worldly_context += scalar * density_matrices[entry.lemma_names()[0]]

    if np.all(abs(worldly_context) < FLOAT_PRECISION):
        return np.identity(DENSITY_DIMENSION)

    return normalize(worldly_context)