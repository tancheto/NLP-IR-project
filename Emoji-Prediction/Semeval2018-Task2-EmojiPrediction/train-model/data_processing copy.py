'''
---- Data processing -----
          ...
'''

import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.parsing.preprocessing import STOPWORDS
from wordsegment import load, segment

texts_file_path = '../data/x-train/train/{}_train.text'
keywords_file_path = '../data/x-train/{}_train.keywords'
train_file_path = '../data/x-train/{}_train.txt'

not_ness_punctuation = re.sub('\?|\!|\.', '', string.punctuation) + '…' + '・'
stop_words = {'us': set(stopwords.words('english')).union(STOPWORDS),
              'es': set(stopwords.words('spanish'))}


def lemmatize_stemming(text, language):
    stemmer = SnowballStemmer(language, ignore_stopwords=True)
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def ala_bala(lang, text):
    language = "spanish" if lang == "es" else 'english'
    text = text.translate(str.maketrans('', '', not_ness_punctuation)).lower()
    return ' '.join([lemmatize_stemming(word, language) for word in word_tokenize(text, language) if word not in stop_words[lang] and len(word) > 3])


def clean_text_vol2(lang, text, mentions=False):
    clean = ""
    keywords = set()
    for word in text.split(" "):
        if not mentions and word == '@':
            break
        elif word.startswith('@') or word == '#':
            pass
        elif word.startswith('#'):
            keywords = keywords.union(segment(word.removeprefix('#')))
        else:
            clean += word + " "

    return [ala_bala(lang, clean.strip()), ' '.join(keywords)]


def data_processing(lang):
    load()

    train = open(train_file_path.format(lang), 'w', encoding="utf8")
    keywords = open(keywords_file_path.format(lang), 'w', encoding="utf8")

    lines = 0

    with open(texts_file_path.format(lang), 'r', encoding="utf8") as texts_file:
        for line in texts_file:
            [text, keywd] = clean_text_vol2(lang, line)

            train.write(text + '\n')
            keywords.write(keywd + '\n')

            lines += 1
            if lines % 10000 == 0 and lines > 0:
                print(str(lines))

            # TO REMOVE
            # if lines == 11:
            #     break

    train.close()
    keywords.close()
