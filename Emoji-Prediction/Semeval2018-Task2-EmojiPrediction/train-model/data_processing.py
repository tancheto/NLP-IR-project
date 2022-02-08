'''
---- Data processing -----
Given a data file with list of pairs - tweet id and origin tweet.
'''

import re
import emojilib

origin_file_path = '../data/train/{}_train.origin'
labels_file_path = '../data/train/{}_train.lables'
texts_file_path = '../data/train/{}_train.text'


def clean_text(text):
    # remove emojis, links and user mentions
    clean = ""
    text = emojilib.replace_emoji(text, replacement=' ')
    text = re.sub('\s+', ' ', text).strip()
    for word in text.split(" "):
        if word.startswith('@') or word.startswith('http'):
            pass
        else:
            clean += word + " "

    # remove double spaces
    return re.sub(' +', ' ', clean).strip()


mapping = {'us': {'❤': '0', '😍': '1', '😂': '2', '💕': '3', '🔥': '4', '😊': '5', '😎': '6', '✨': '7', '💙': '8', '😘': '9', '📷': '10', '🇺🇸': '11', '☀': '12', '💜': '13', '😉': '14', '💯': '15', '😁': '16', '🎄': '17', '📸': '18', '😜': '19'},
           'es': {'❤': '0', '😍': '1', '😂': '2', '💕': '3', '😊': '4', '😘': '5', '💪': '6', '😉': '7', '👌': '8', '🇪🇸': '9', '😎': '10', '💙': '11', '💜': '12', '😜': '13', '💞': '14', '✨': '15', '🎶': '16', '💘': '17', '😁': '18', '	': '19'}}


def data_processing(lang):
    labels = open(labels_file_path.format(lang), 'w', encoding="utf8")
    texts = open(texts_file_path.format(lang), 'w', encoding="utf8")

    number = 10

    with open(origin_file_path.format(lang), encoding="utf8") as origin_file:
        for line in origin_file:
            number -= 1
            if number <= 0:
                break

            text = line.split('\t')[1]
            emo_list = emojilib.emoji_list(text)
            emo_set = set([d['code'] for d in emo_list if 'code' in d])

            if len(emo_set) > 0:
                for emoji in emo_set:
                    if emoji in mapping[lang]:
                        labels.write(emoji + ' ')

                labels.write('\n')
                texts.write(clean_text(text) + '\n')
