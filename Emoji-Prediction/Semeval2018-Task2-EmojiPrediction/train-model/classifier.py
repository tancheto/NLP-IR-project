'''
---- Classifier -----
          ...
'''

import re
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
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

# predictions path
predictions_file_path = '../data/x-train/predictions/{}_predictions_{}.labels'

mask_file_path = '../data/x-train/masks/{}_{}_mask.txt'


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
        words, keywd = preprocessing(lang, line)
        processed_data.append(words + keywd)
        lbls.append(re.sub('\n', '', lbl))

        rows += 1
        if rows == n:
            break

    text.close()
    labels.close()

    return processed_data, lbls


def load_data_and_write_masks(lang):
    print("train data loading ...")
    train_data = load_data(lang, train_file_path)

    print("trial data loading ...")
    trial_data = load_data(lang, trial_file_path)

    print("test data loading ...")
    test_data = load_data(lang, test_file_path)

    features_train, all_mappings_train = get_features(train_data, [1])
    features_trial, all_mappings_trial = get_features(trial_data, [1])
    features_test, all_mappings_test = get_features(test_data, [1])

    # dictionary, BoW = get_dictionary(train_data, 0)
    # tfidf = tf_idf(BoW)

    train = get_binary_representation(features_train, all_mappings_train)
    write_mask_in_file(train, mask_file_path.format(lang, "train"))

    trial = get_binary_representation(features_train, all_mappings_trial)
    write_mask_in_file(trial, mask_file_path.format(lang, "trial"))

    test = get_binary_representation(features_train, all_mappings_test)
    write_mask_in_file(test, mask_file_path.format(lang, "test"))


def classification(lang):
    load_data_and_write_masks(lang)

    number = 100

    train = read_mask_from_file(mask_file_path.format(lang, "train"))[:number]
    trial = read_mask_from_file(mask_file_path.format(lang, "trial"))[:number]
    test = read_mask_from_file(mask_file_path.format(lang, "test"))[:number]

    train_labels = load_labels(lang, train_labels_file_path)[:number]
    trial_labels = load_labels(lang, trial_labels_file_path)[:number]
    test_labels = load_labels(lang, test_labels_file_path)[:number]

    # classifier = RandomForestClassifier()
    classifier = svm.SVC()
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
