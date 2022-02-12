'''
---- Classifier -----
          ...
'''

from cProfile import label
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import preprocessing
from feature_engineering import *

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


def load_labels(lang, path):
    with open(path.format(lang), 'r', encoding="utf8") as labels:
        lbls = []
        for line in labels:
            lbls.append(re.sub('\n', '', line))
    return lbls


def load_data(lang, path):
    with open(path.format(lang), 'r', encoding="utf8") as data:
        processed_data = []
        for line in data:
            text, keywd = preprocessing(lang, line)
            processed_data.append(text + keywd)
    return processed_data


def load_data_and_labels(lang, path, n):
    text = open((path + ".text").format(lang), 'r', encoding="utf8")
    labels = open((path + ".labels").format(lang), 'r', encoding="utf8")

    lbls = []
    processed_data = []

    rows = 0
    for line, lbl in zip(text, labels):
        text, keywd = preprocessing(lang, line)
        processed_data.append(text + keywd)
        lbls.append(re.sub('\n', '', lbl))

        rows += 1
        if rows == n:
            break

    return processed_data, lbls


def classification(lang):
    n = 3

    print("train data loading ...")
    train_data, train_labels = load_data_and_labels(lang, train_file_path, n)

    print("trial data loading ...")
    trial_data, trial_labels = load_data_and_labels(lang, trial_file_path, n)

    print("test data loading ...")
    test_data, test_labels = load_data_and_labels(lang, test_file_path, n)

    # features, all_mappings = get_features(train_data, [1, 2, 3])
    # dictionary, BoW = get_dictionary(train_data, 0)
    # tfidf = tf_idf(BoW)

    classifier = RandomForestClassifier()
    print("learning ...")
    # classifier.fit(train_data, train_labels)
    print("evaluating ...")
    # classifier.score(trial_data, trial_labels)

    predicted = -1
    # predicted = classifier.predict(test_data[0])
    print(f"predicted: {predicted}, real: {test_labels[0]}")

    print("Step 4: 💯%")
