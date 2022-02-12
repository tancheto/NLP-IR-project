'''
---- Feature Engineering -----
          ...
'''

import re
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

# paths
train_file_path = '../data/x-train/{}_train.txt'
keywords_file_path = '../data/x-train/{}_train.keywords'

features_file_path = '../data/x-train/processed/{}_features.txt'
row_ngrams_file_path = '../data/x-train/processed/{}_row_ngrams.txt'

dictionary_file_path = '../data/x-train/processed/{}_dictionary.txt'
bag_of_words_file_path = '../data/x-train/processed/{}_bag_of_words.txt'

inverted_index_file_path = '../data/x-train/processed/{}_inverted_index.txt'
tfidf_file_path = '../data/x-train/processed/{}_tfidf.txt'

# important variables
most_important_ngrams = 1000


def get_preprocessed_data(lang):
    # load train data
    train = open(train_file_path.format(lang), 'r', encoding="utf8")
    keywords = open(keywords_file_path.format(lang), 'r', encoding="utf8")

    all_sentences = []
    for line, line_kw in zip(train, keywords):
        union = re.sub('\n', '', line + ' ' + line_kw)
        all_sentences.append(union.strip().split(' '))
    return all_sentences


def get_dictionary(processed_doc, keep_n):
    # dictionary of words and their frequencies (n most frequent)
    dictionary = {}
    if keep_n == 0:
        dictionary = Dictionary(processed_doc)
        dictionary.filter_extremes(no_below=3, keep_n=None)
    else:
        dictionary = Dictionary(processed_doc, keep_n)
        dictionary.filter_extremes(no_below=3, keep_n=keep_n)

    # document in Bag of Words(BoW) preview (using dictionary)
    BoW_doc = [dictionary.doc2bow(row, False) for row in processed_doc]

    return dictionary, BoW_doc


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


def get_most_frequent_ngrams(ngram_frequencies, keep_n):
    # ngram sort and culling based on the frequencies
    s_ngrams = sorted(ngram_frequencies.items(),
                      key=lambda item: item[1], reverse=True)
    return list(map(lambda x: x[0], s_ngrams))[0:keep_n]


def get_features(data, ns_in_ngrams):
    # get most frequent ngrams (for different ns)
    # get for every data row all ngrams (for thise ns)
    features = []
    all_mappings = {}

    # initialising
    for idx in range(1, len(data) + 1):
        all_mappings[idx] = []

    # find ngrams for different ns
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


def tf_idf(bag_of_words):
    tfidf_model = []
    tfidf = TfidfModel(bag_of_words)
    for row in tfidf[bag_of_words]:
        tfidf_model.append(row)
    return tfidf_model


def inverted_index(dictionary, processed_doc):
    # create for every word in dictionary list of row indexes of its encounters

    # initialising with empty lists
    inverted_idx = {}
    for key, value in dictionary.iteritems():
        inverted_idx[value] = []

    dictionary_set = set(dictionary.values())

    # create the inverted index
    for idx, row in enumerate(processed_doc):
        for word in row:
            if word in dictionary_set:
                inverted_idx[word].append(str(idx + 1))
    return inverted_idx


def Dict_BoW_IIDX(lang, processed_doc):
    # create dictionary, bag of words and inverted index from processed document
    dictionary, BoW = get_dictionary(processed_doc, 0)

    print("dictionary ...")
    dict = open(dictionary_file_path.format(lang), 'w', encoding="utf8")
    for key, value in dictionary.iteritems():
        dict.write(f"{key}:{value}\n")

    print("bag of words ...")
    bag = open(bag_of_words_file_path.format(lang), 'w', encoding="utf8")
    for row in BoW:
        bag.write(' '.join([f"{tuple[0]}-{tuple[1]}" for tuple in row]) + '\n')

    print("inverted index ...")
    inv_idx = open(inverted_index_file_path.format(lang), 'w', encoding="utf8")
    indexes = inverted_index(dictionary, processed_doc)
    for word in indexes:
        inv_idx.write(f"{word}:{' '.join(indexes[word])}\n")

    print("tf-idf ...")
    tfidf = open(tfidf_file_path.format(lang), 'w', encoding="utf8")
    for row in tf_idf(BoW):
        tfidf.write(' '.join([f"{word[0]}-{word[1]}" for word in row]) + '\n')

    dict.close()
    bag.close()
    inv_idx.close()
    tfidf.close()


def feature_engineering(lang, processed_doc):
    # find ngrams of one, two and three words
    features, all_mappings = get_features(processed_doc, [1, 2, 3])

    print("features ...")
    feats = open(features_file_path.format(lang), 'w', encoding="utf8")
    for feature in features:
        feats.write(feature + '\n')

    print("row ngrams ...")
    row_ngrams = open(row_ngrams_file_path.format(lang), 'w', encoding="utf8")
    for idx in all_mappings:
        row_ngrams.write('|'.join(all_mappings[idx]) + '\n')

    feats.close()
    row_ngrams.close()


def write_mask_in_file(mask, path):
    with open(path, 'w', encoding="utf8") as mask_file:
        for row in mask:
            mask_file.write(' '.join([str(item) for item in row]) + '\n')


def read_mask_from_file(path):
    mask = []
    with open(path, 'r', encoding="utf8") as mask_file:
        for row in mask:
            mask.append(re.sub('\n', '', row).split(' '))
    return mask


def all(lang):
    # train data
    data_rows = get_preprocessed_data(lang)

    Dict_BoW_IIDX(lang, data_rows)
    feature_engineering(lang, data_rows)

    print("Step 3: 💯%")
