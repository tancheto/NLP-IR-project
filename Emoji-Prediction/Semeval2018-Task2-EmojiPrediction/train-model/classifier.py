'''
---- Classifier -----
          ...
'''

import re
import time
import config
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from data_preprocessing import preprocessing
from feature_engineering import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# train paths
train_labels_file_path = '../data/x-train/train/{}_train.labels'
train_text_file_path = '../data/x-train/train/{}_train.text'
train_file_path = '../data/x-train/train/{}_train'

# trial paths
trial_labels_file_path = '../data/trial/{}_trial.labels'
trial_text_file_path = '../data/trial/{}_trial.text'
trial_file_path = '../data/trial/{}_trial'

# test paths
test_labels_file_path = '../data/test/{}_test.labels'
test_text_file_path = '../data/test/{}_test.text'
test_file_path = '../data/test/{}_test'

# masks paths
mask_file_path = '../data/x-train/processed/masks/{}_{}_mask.txt'

# predictions path
predictions_file_path = '../data/x-train/predictions/{}_predictions_{}.labels'

# important variables
loaded = config.classifier_load
ngrams_types = config.ngrams_types
most_important_ngrams = config.most_important_ngrams


def load_labels(lang, path):
    with open(path.format(lang), 'r', encoding="utf8") as labels:
        lbls = []
        rows = 0
        for line in labels:
            lbls.append(re.sub('\n', '', line))

            rows += 1
            if rows == loaded:
                break
    return lbls


def load_data(lang, path):
    with open(path.format(lang), 'r', encoding="utf8") as data:
        processed_data = []
        rows = 0
        for line in data:
            text, keywd = preprocessing(lang, line)
            processed_data.append(' '.join(text + keywd))

            rows += 1
            if rows == loaded:
                break
    return processed_data


def load_data_and_labels(lang, path):
    text = open((path + ".text").format(lang), 'r', encoding="utf8")
    labels = open((path + ".labels").format(lang), 'r', encoding="utf8")

    lbls = []
    processed_data = []

    rows = 0
    for line, lbl in zip(text, labels):
        words, keywd = preprocessing(lang, line)
        processed_data.append(words + keywd)
        lbls.append(re.sub('\n', '', lbl))

        rows += 1
        if rows == loaded:
            break

    text.close()
    labels.close()

    return processed_data, lbls


def load_data_and_write_masks_1(lang):
    print("train data loading ...")
    train_data = load_data(lang, train_text_file_path)

    print("trial data loading ...")
    trial_data = load_data(lang, trial_text_file_path)

    print("test data loading ...")
    test_data = load_data(lang, test_text_file_path)

    features_train, all_mappings_train = get_features(
        train_data, ngrams_types, most_important_ngrams)
    features_trial, all_mappings_trial = get_features(
        trial_data, ngrams_types, most_important_ngrams)
    features_test, all_mappings_test = get_features(
        test_data, ngrams_types, most_important_ngrams)

    print("creating train mask ...")
    train = get_binary_representation(features_train, all_mappings_train)
    write_mask_in_file(train, mask_file_path.format(lang, "train"))

    print("creating trial mask ...")
    trial = get_binary_representation(features_train, all_mappings_trial)
    write_mask_in_file(trial, mask_file_path.format(lang, "trial"))

    print("creating test mask ...")
    test = get_binary_representation(features_train, all_mappings_test)
    write_mask_in_file(test, mask_file_path.format(lang, "test"))


def load_data_and_write_masks_2(lang):
    print("train data loading ...")
    train_data = load_data(lang, train_text_file_path)

    print("trial data loading ...")
    trial_data = load_data(lang, trial_text_file_path)

    print("test data loading ...")
    test_data = load_data(lang, test_text_file_path)

    dictionary_train, BoW_train = get_dictionary(train_data, 0)
    tfidf_train = tf_idf(BoW_train)

    dictionary_trial, BoW_trial = get_dictionary(trial_data, 0)
    tfidf_trial = tf_idf(BoW_trial)

    dictionary_test, BoW_test = get_dictionary(test_data, 0)
    tfidf_test = tf_idf(BoW_test)

    print("creating train mask ...")
    train = get_mask(dictionary_train, tfidf_train)
    write_mask_in_file(train, mask_file_path.format(lang, "train"))

    print("creating trial mask ...")
    trial = get_mask(dictionary_train, tfidf_trial)
    write_mask_in_file(trial, mask_file_path.format(lang, "trial"))

    print("creating test mask ...")
    test = get_mask(dictionary_train, tfidf_test)
    write_mask_in_file(test, mask_file_path.format(lang, "test"))


def load_data_and_write_masks_3(lang):
    print("train data loading ...")
    train_data = load_data(lang, train_text_file_path)

    print("trial data loading ...")
    trial_data = load_data(lang, trial_text_file_path)

    print("test data loading ...")
    test_data = load_data(lang, test_text_file_path)

    features_train, all_mappings_train = get_features(
        train_data, ngrams_types, most_important_ngrams)
    features_trial, all_mappings_trial = get_features(
        trial_data, ngrams_types, most_important_ngrams)
    features_test, all_mappings_test = get_features(
        test_data, ngrams_types, most_important_ngrams)

    dictionary_train, BoW_train = get_dictionary(train_data, 0)
    tfidf_train = tf_idf(BoW_train)

    dictionary_trial, BoW_trial = get_dictionary(trial_data, 0)
    tfidf_trial = tf_idf(BoW_trial)

    dictionary_test, BoW_test = get_dictionary(test_data, 0)
    tfidf_test = tf_idf(BoW_test)

    print("creating train mask ...")
    train = get_mask(dictionary_train, tfidf_train)
    write_mask_in_file(train, mask_file_path.format(lang, "train"))

    print("creating trial mask ...")
    trial = get_binary_representation(dictionary_train, all_mappings_trial)
    write_mask_in_file(trial, mask_file_path.format(lang, "trial"))

    print("creating test mask ...")
    test = get_binary_representation(dictionary_train, all_mappings_test)
    write_mask_in_file(test, mask_file_path.format(lang, "test"))


def load_data_and_write_masks_4(lang):
    print("train data loading ...")
    train_data = load_data(lang, train_text_file_path)

    print("trial data loading ...")
    trial_data = load_data(lang, trial_text_file_path)

    print("test data loading ...")
    test_data = load_data(lang, test_text_file_path)

    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features=3000)

    print("creating train mask ...")
    tfidfvectorizer.fit(train_data)
    train = tfidfvectorizer.transform(train_data).toarray()
    write_mask_in_file(train, mask_file_path.format(lang, "train"))

    print("creating trial mask ...")
    trial = tfidfvectorizer.transform(trial_data).toarray()
    write_mask_in_file(trial, mask_file_path.format(lang, "trial"))

    print("creating test mask ...")
    test = tfidfvectorizer.transform(test_data).toarray()
    write_mask_in_file(test, mask_file_path.format(lang, "test"))


def classification(lang):
    if config.has_to_load:
        # chose from 4 different variants ...
        load_data_and_write_masks_4(lang)

    train = read_mask_from_file(mask_file_path.format(lang, "train"))[:loaded]
    trial = read_mask_from_file(mask_file_path.format(lang, "trial"))[:loaded]
    test = read_mask_from_file(mask_file_path.format(lang, "test"))[:loaded]

    train_labels = load_labels(lang, train_labels_file_path)[:loaded]
    trial_labels = load_labels(lang, trial_labels_file_path)[:loaded]
    test_labels = load_labels(lang, test_labels_file_path)[:loaded]

    classifier = RandomForestClassifier()
    # classifier = svm.SVC()
    print("learning ...")
    classifier.fit(train, train_labels)
    print("evaluating ...")
    classifier.score(trial, trial_labels)
    print("predicting ...")
    predicted = classifier.predict(test)

    with open(predictions_file_path.format(lang, time.time()), 'w', encoding="utf8") as predic:
        for label in predicted:
            predic.write(label + '\n')

    # precision = precision_score(test_labels, predicted, average = "macro")
    # recall = recall_score(test_labels, predicted, average = "macro")
    f1 = f1_score(test_labels, predicted, average="micro")
    # print(f"F1: {f1}, Precision: {precision}, Recall: {recall}")

    print(f"Step 4: 💯% -> F1: {f1}")
    return test_labels, predicted
