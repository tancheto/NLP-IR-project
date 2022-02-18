'''
---- Classifier -----
          ...
'''

import re
import time
import config
from sklearn.metrics import f1_score
from data_preprocessing import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from mask_creator import *
from visualisation import visualisation


# train paths
train_labels_file_path = '../data/x-train/train/{}_train.labels'
train_text_file_path = '../data/x-train/train/{}_train.text'

# trial paths
trial_labels_file_path = '../data/trial/{}_trial.labels'
trial_text_file_path = '../data/trial/{}_trial.text'

# test paths
test_labels_file_path = '../data/test/{}_test.labels'
test_text_file_path = '../data/test/{}_test.text'

# masks paths
mask_file_path = '../data/x-train/processed/masks/{}_{}_mask.txt'

# idf features
idf_features_file_path = '../data/x-train/processed/{}_idf_features.txt'

# predictions path
predictions_file_path = '../data/x-train/predictions/{}_predictions_{}.labels'

# important variables
loaded = config.classifier_load
ngrams_types = config.ngrams_types
min_frequency = config.min_frequency
max_features = config.max_features
ngram_range = config.ngram_range


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


def load_data_and_write_masks(lang):
    print("train data loading ...")
    train_data = load_data(lang, train_text_file_path)

    # print("trial data loading ...")
    # trial_data = load_data(lang, trial_text_file_path)

    print("test data loading ...")
    test_data = load_data(lang, test_text_file_path)

    print("calculating idf ...")
    tfidfvectorizer = TfidfVectorizer(
        analyzer='word', stop_words='english', ngram_range=ngram_range, max_features=max_features, min_df=min_frequency)
    tfidfvectorizer.fit(train_data)

    idf_features = tfidfvectorizer.get_feature_names_out()
    with open(idf_features_file_path.format(lang), 'w', encoding="utf8") as idf_features_file:
        for feature in idf_features:
            idf_features_file.write(feature + '\n')

    print("creating train mask ...")
    train = tfidfvectorizer.transform(train_data).toarray()
    write_mask_in_file(train, mask_file_path.format(lang, "train"))

    # print("creating trial mask ...")
    # trial = tfidfvectorizer.transform(trial_data).toarray()
    # write_mask_in_file(trial, mask_file_path.format(lang, "trial"))

    print("creating test mask ...")
    test = tfidfvectorizer.transform(test_data).toarray()
    write_mask_in_file(test, mask_file_path.format(lang, "test"))


def classification(lang):
    if config.has_to_load:
        load_data_and_write_masks(lang)

    train = read_mask_from_file(mask_file_path.format(lang, "train"))[:loaded]
    # trial = read_mask_from_file(mask_file_path.format(lang, "trial"))[:loaded]
    test = read_mask_from_file(mask_file_path.format(lang, "test"))[:loaded]

    train_labels = load_labels(lang, train_labels_file_path)[:loaded]
    # trial_labels = load_labels(lang, trial_labels_file_path)[:loaded]
    test_labels = load_labels(lang, test_labels_file_path)[:loaded]

    # classifier = RandomForestClassifier()
    # classifier = LogisticRegression()
    classifier = KNeighborsClassifier()
    print("learning ...")
    classifier.fit(train, train_labels)
    # print("evaluating ...")
    # classifier.score(trial, trial_labels)
    print("predicting ...")
    predicted = classifier.predict(test)

    timestamp = time.time()
    predicted_file = predictions_file_path.format(lang, timestamp)
    with open(predicted_file, 'w', encoding="utf8") as predic:
        for label in predicted:
            predic.write(label + '\n')

    print("result visualisation ...")
    gold_file_path = test_labels_file_path.format(lang)
    visualisation(lang, gold_file_path, predicted_file)

    f1_micro = f1_score(test_labels, predicted, average="micro")
    f1_macro = f1_score(test_labels, predicted, average="macro")
    print(
        f"Step 4: 💯% -> F1-macro: {round(f1_macro, 3)} & F1-micro: {round(f1_micro, 3)}")
