'''
---- Feature Engineering -----
          ...
'''

import re

# paths
train_file_path = '../data/x-train/{}_train.txt'

# important variables
most_important_ngrams = 1000


def get_preprocessed_data(lang):
    # load train data
    all_sentences = []
    with open(train_file_path.format(lang), 'r', encoding="utf8") as train:
        for line in train:
            all_sentences.append(re.sub('\n', '', line).split(' '))
    return all_sentences


def get_ngrams(n, sentences):
    # find ngrams and their frequencies
    ngram_frequencies = {}
    idx_ngrams_mapping = {}
    for idx, sentence in enumerate(sentences):
        ngrams_sentence = []
        for i in range(len(sentence) - n + 1):
            ngram = ' '.join(sentence[i:i+n])
            ngrams_sentence.append(ngram)
            # either the frequency, or 0
            freq = ngram_frequencies.get(ngram, 0)
            ngram_frequencies[ngram] = freq + 1
        idx_ngrams_mapping[idx + 1] = ngrams_sentence
    return ngram_frequencies, idx_ngrams_mapping


def get_most_frequent_ngrams(ngram_frequencies, number):
    # ngram sort and culling based on the frequencies
    s_ngrams = sorted(ngram_frequencies.items(),
                      key=lambda item: item[1], reverse=True)
    return list(map(lambda x: x[0], s_ngrams))[0:number]


def get_features(data, ns_in_ngrams):
    features = []
    all_mappings = {}

    for idx in range(1, len(data) + 1):
        all_mappings[idx] = []

    for n in ns_in_ngrams:
        ngrams, idx_ngrams_mapping = get_ngrams(n, data)
        curr_features = get_most_frequent_ngrams(ngrams, most_important_ngrams)
        features += curr_features
        for idx in idx_ngrams_mapping:
            all_mappings[idx] += idx_ngrams_mapping[idx]
    return features, all_mappings


def get_binary_representation(features, mappings):
    # create binary mask of occurrences
    binary_representation = []
    for key in mappings:
        curr_representation = [
            1 if feature in mappings[key] else 0 for feature in features]
        binary_representation.append(curr_representation)
    return binary_representation


def feature_engineering(lang):
    # train data
    data_rows = get_preprocessed_data(lang)

    # find ngrams of one, two and three words
    features, all_mappings = get_features(data_rows, [1, 2, 3])

    fs = open('../data/x-train/{}_fs.txt'.format(lang), 'w', encoding="utf8")
    for feature in features:
        fs.write(feature + '\n')

    mp = open('../data/x-train/{}_mp.txt'.format(lang), 'w', encoding="utf8")
    for idx in all_mappings:
        mp.write('|'.join(all_mappings[idx]) + '\n')

    # get mask
    data_rows_binary = get_binary_representation(features, all_mappings)

    # DO NOT PRINT
    # print(data_rows_binary)
