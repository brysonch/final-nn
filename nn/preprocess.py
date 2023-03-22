# Imports
import numpy as np
import random
from typing import List, Tuple
from numpy.typing import ArrayLike
from random import choices

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    
    pos_ind = [i for i, b in enumerate(labels) if b == 1]
    neg_ind = [i for i, b in enumerate(labels) if b == 0]

    pos_seqs = [seqs[i] for i in pos_ind]
    neg_seqs = [seqs[i] for i in neg_ind]

    num_pos = len(pos_seqs)
    num_neg = len(neg_seqs)

    if num_pos < num_neg:

        resampled_pos = choices(pos_seqs, k=num_neg)
        unshuffled_seqs = resampled_pos + neg_seqs
        unshuffled_labels = [1] * num_neg + [0] * num_neg

        combined = list(zip(unshuffled_seqs, unshuffled_labels))
        random.shuffle(combined)
        resampled_seqs, resampled_labels = zip(*combined)

        return (resampled_seqs, resampled_labels)
    elif num_pos > num_neg:

        resampled_neg = choices(pos_seqs, k=num_pos)
        unshuffled_seqs = pos_seqs + resampled_neg
        unshuffled_labels = [1] * num_pos + [0] * num_pos

        combined = list(zip(unshuffled_seqs, unshuffled_labels))
        random.shuffle(combined)
        resampled_seqs, resampled_labels = zip(*combined)

        return (resampled_seqs, resampled_labels)

    combined = list(zip(pos_seqs, neg_seqs))
    random.shuffle(combined)
    resampled_seqs, resampled_labels = zip(*combined) 

    return (resampled_seqs, resampled_labels)


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
    }

    encodings = [np.hstack(list(map(mapping.get, seq))) for seq in seq_arr]
    return encodings
