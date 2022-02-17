'''
---- Mask Creator -----
          ...
'''

import re

# paths
# important variables


def get_binary_representation(features, mappings):
    # create binary mask of occurrences
    binary_representation = []
    for key in mappings:
        curr_representation = [
            1 if feature in mappings[key] else 0 for feature in features]
        binary_representation.append(curr_representation)
    return binary_representation


def tfidf_mask(dictionary, tfidf_model):
    mask = []
    for row in tfidf_model:
        words = dict(row)
        curr_representation = [
            words[idx] if idx in words.keys() else 0 for idx, word in dictionary.items()]
        mask.append(curr_representation)
    return mask


def write_mask_in_file(mask, path):
    with open(path, 'w', encoding="utf8") as mask_file:
        for row in mask:
            mask_file.write(' '.join([str(item) for item in row]) + '\n')


def read_mask_from_file(path):
    mask = []
    with open(path, 'r', encoding="utf8") as mask_file:
        for row in mask_file:
            mask.append(re.sub('\n', '', row).split(' '))
    return mask
