'''
---- Data processing -----
Given a data file with list of pairs - tweet id and origin tweet.
Return ... files.
'''

import re
import emojilib
from nltk.corpus import stopwords

origin_file_path = '../data/train/{}_train.origin'
texts_file_path = '../data/train/{}_train.text'
labels_file_path = '../data/train/{}_train.lables'
keywords_file_path = '../data/train/{}_train.keywords'

stop_words = {'us': set(stopwords.words('english')),
              'es': set(stopwords.words('spanish'))}


def clean_text_vol2(lang, text):
    clean = ""
    keywords = set()

    text = emojilib.replace_emoji(text, replacement=' ')
    text = re.sub('\s+', ' ', text).strip()

    text = text.lower()

    for word in text.split(" "):
        if word.startswith('@'):
            break
        elif word in stop_words[lang] or word.startswith('http') or (word.startswith('#') and len(word) == 1):
            pass
        elif word.startswith('#'):
            keywords.add(word.removeprefix('#'))
        else:
            clean += word + " "

    return [re.sub('\s+', ' ', clean).strip(), keywords]


def data_processing(lang):
    texts = open(texts_file_path.format(lang), 'w', encoding="utf8")
    labels = open(labels_file_path.format(lang), 'w', encoding="utf8")
    keywords = open(keywords_file_path.format(lang), 'w', encoding="utf8")

    number = 10

    with open(origin_file_path.format(lang), 'r', encoding="utf8") as origin_file:
        for line in origin_file:
            number -= 1
            if number < 0:
                break

            text = line.split('\t')[1]

            emo_list = emojilib.emoji_list(text)
            emo_set = set([d['code'] for d in emo_list if 'code' in d])
            labels.write(' '.join(emo_set) + '\n')

            [clean, keyws] = clean_text_vol2(lang, text)
            keywords.write(' '.join(keyws) + '\n')
            texts.write(clean + '\n')
