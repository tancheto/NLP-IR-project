'''
---- Raw data processing -----
Given a raw data file with one tweet per line.
Return one file with list of pairs - tweet id and origin tweet, that are not in test and trial.
'''

import re
import json
import emojilib

raw_data = '../data/train/raw/{}_raw_data.txt'
test_file_path = '../data/test/{}_test.text'
trial_file_path = '../data/trial/{}_trial.text'
train_file_path = '../data/train/{}_train.origin'


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
    # remove double spaces
    return clean.strip()


def raw_data_processing(lang):
    test = open(test_file_path.format(lang), 'r', encoding="utf8")
    trial = open(trial_file_path.format(lang), 'r', encoding="utf8")
    out_origin = open(train_file_path.format(lang), 'w', encoding="utf8")

    test_trial = set()

    for line_test in test:
        test_trial.add(line_test.strip())

    for line_trial in trial:
        test_trial.add(line_trial.strip())

    total = 0
    good = 0
    unique = 0
    removed = 0

    with open(raw_data.format(lang), 'r', encoding="utf8") as file:
        for line in file:
            json_line = json.loads(line)

            tweet_id = json_line['id']
            text = json_line['text'].replace("\n", "")

            emo_list = emojilib.emoji_list(text)
            emo_set = set([d['code'] for d in emo_list if 'code' in d])

            if len(emo_set) > 0 and all(emo in emojilib.mapping[lang] for emo in emo_set):
                if clean_text(text) in test_trial:
                    removed += 1
                else:
                    out_origin.write(str(tweet_id) + '\t' + text + "\n")
                    unique += 1
                good += 1
            total += 1

            if total % 10000 == 0 and total > 0:
                print(str(total))

    print(f"Total: {total}, Good: {good}, Unique: {unique}, Removed: {removed}")

    test.close()
    trial.close()
    out_origin.close()
