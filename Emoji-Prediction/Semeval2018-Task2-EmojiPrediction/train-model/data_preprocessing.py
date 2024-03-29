'''
---- Data preprocessing -----
          ...
'''

import string
import config
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.parsing.preprocessing import STOPWORDS
from wordsegment import load, segment

# loading the data for wordsegment
load()

# paths
texts_file_path = '../data/x-train/train/{}_train.text'
keywords_file_path = '../data/x-train/{}_train.keywords'
train_file_path = '../data/x-train/{}_train.txt'

# important variables
add_mentions = config.add_mentions
unnecess_punctuation = string.punctuation + '…' + '・' + '•'
stop_words = {'us': set(stopwords.words('english')).union(STOPWORDS),
              'es': set(stopwords.words('spanish'))}


def lemmatize_stemming(text, language, pos='n'):
    # stemming and lemmatizing the text
    stemmer = SnowballStemmer(language, ignore_stopwords=True)
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos))
    # pos: n(nouns), v(verbs), (a)adjectives, r(adverbs), s(satellite adjectives)


def sentence_tokenizer(lang, text):
    # remove punctuation, tokenize and lemmatize
    language = "spanish" if lang == "es" else 'english'
    text = text.translate(str.maketrans('', '', unnecess_punctuation)).lower()
    return [lemmatize_stemming(word, language) for word in word_tokenize(text, language) if word not in stop_words[lang] and len(word) > 2]


def clean_text_vol2(text, mentions=False):
    # remove mentions and extract hashtags (works only for english)
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

    return clean.strip(), ' '.join(keywords)


def preprocessing(lang, text):
    # one tweet preprocessing
    clean_text, keywd = clean_text_vol2(text, add_mentions)
    clean_text = sentence_tokenizer(lang, clean_text)
    keywd = sentence_tokenizer(lang, keywd)
    return clean_text, keywd


def data_preprocessing(lang):
    # open files to write
    train = open(train_file_path.format(lang), 'w', encoding="utf8")
    keywords = open(keywords_file_path.format(lang), 'w', encoding="utf8")

    lines = 0
    with open(texts_file_path.format(lang), 'r', encoding="utf8") as texts_file:
        for line in texts_file:
            text, keywd = preprocessing(lang, line)

            train.write(' '.join(text) + '\n')
            keywords.write(' '.join(keywd) + '\n')

            lines += 1
            if lines % 10000 == 0 and lines > 0:
                print(str(lines))

    # close files
    train.close()
    keywords.close()

    print("Step 2: 💯%")
