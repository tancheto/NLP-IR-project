'''
---- Raw data processing -----
Given a raw data file with one tweet per line.
Return one file with list of pairs - tweet id and origin tweet, that are not in test and trial.
'''

import re
import json
import libs.emoji.emojilib as emojilib

# paths
test_file_path = '../data/test/{}_test.text'
trial_file_path = '../data/trial/{}_trial.text'
raw_file_path = '../data/x-train/raw/{}_raw_data.txt'
origin_file_path = '../data/x-train/raw/{}_train.origin'

texts_file_path = '../data/x-train/train/{}_train.text'
labels_file_path = '../data/x-train/train/{}_train.lables'


def clean_text(text):
    # remove emojis, remove links, anonymize user mentions
    clean = ""
    text = emojilib.replace_emoji(text, replacement=' ')
    text = re.sub('\s+', ' ', text).strip()
    for word in text.split(" "):
        if word.startswith('@') and len(word) > 1:
            clean += "@user "
        elif word.startswith('http'):
            pass
        else:
            clean += word + " "
    return clean.strip()


def raw_data_processing(lang):
    # open files to read
    test = open(test_file_path.format(lang), 'r', encoding="utf8")
    trial = open(trial_file_path.format(lang), 'r', encoding="utf8")

    # open files to write
    out_origin = open(origin_file_path.format(lang), 'w', encoding="utf8")
    labels = open(labels_file_path.format(lang), 'w', encoding="utf8")
    texts = open(texts_file_path.format(lang), 'w', encoding="utf8")

    # union of trial and test data
    test_trial = set()

    for line_test in test:
        test_trial.add(line_test.strip())

    for line_trial in trial:
        test_trial.add(line_trial.strip())

    total = 0
    good = 0
    unique = 0
    removed = 0

    # open raw data file
    with open(raw_file_path.format(lang), 'r', encoding="utf8") as file:
        for line in file:
            # extract tweet 'id' and 'text' from json line
            json_line = json.loads(line)
            tweet_id = json_line['id']
            text = json_line['text']

            # remove unnecessary white spaces
            text = re.sub('\n|\r|\r\n|\n\r', ' ', text)
            text = re.sub('\s+', ' ', text).strip()

            # extract emojies from text
            emo_list = emojilib.emoji_list(text)
            emo_set = set([d['code'] for d in emo_list if 'code' in d])

            # only tweets with emojies from given set
            if len(emo_set) > 0 and all(emo in emojilib.mapping[lang] for emo in emo_set):
                clean = clean_text(text)
                # check if the tweet is in test or trial
                if clean in test_trial:
                    removed += 1
                else:
                    out_origin.write(str(tweet_id) + '\t' + text + '\n')
                    labels.write(' '.join(emo_set) + '\n')
                    texts.write(clean + '\n')
                    unique += 1
                good += 1
            total += 1

            if total % 10000 == 0 and total > 0:
                print(str(total))

            # TO REMOVE
            if total == 11111:
                break

    print(f"Total: {total}, Good: {good}, Unique: {unique}, Removed: {removed}")

    # close files
    test.close()
    trial.close()
    out_origin.close()
    labels.close()
    texts.close()
