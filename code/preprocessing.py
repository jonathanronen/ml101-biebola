"""
For Intro to Data Science, NYU 2014.

Reads raw csv file, cleans texts and counts words in dictionary.

@auth jonathanronen
"""

import string
from collections import defaultdict
# available on https://github.com/SMAPPNYU/smappPy
from smappPy.unicode_csv import UnicodeReader, UnicodeWriter
from nltk.corpus import stopwords
import unicodedata
import sys

TWEETS_FILENAME = 'data/tweets.csv'
CLEAN_DATA_FILENAME = 'data/clean_data.csv'
DICTIONARY_COUNT_FILENAME = 'data/dictionary.csv'



tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))
def remove_punctuation(text):
    return text.translate(tbl)

def clean_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    text = ' '.join([word for word in text.split() if not word.isnumeric()])
    text = ' '.join([word for word in text.split() if not 'bieber' in word and not 'ebola' in word])
    text = text.replace('^', '')
    text = text.replace('=', '')
    text = text.replace('+', '')
    return text


if __name__ == '__main__':

    tweet_texts = list()
    tweet_labels = list()

    with open(TWEETS_FILENAME, 'rb') as input_file:
        csv_reader = UnicodeReader(input_file)
        HEADER = csv_reader.next()
        TEXT_INDEX = HEADER.index('text')
        LABEL_INDEX = HEADER.index('label')

        for line in csv_reader:
            label = line[LABEL_INDEX]
            text  = line[TEXT_INDEX]
            clean = clean_text(text)

            tweet_texts.append(clean)
            tweet_labels.append(label)
            sys.stdout.write('+')
        sys.stdout.write('\n')

    dictionary = defaultdict(lambda: 0)
    for text in tweet_texts:
        for word in text.split():
            dictionary[word] += 1
        sys.stdout.write('.')

    with open(CLEAN_DATA_FILENAME, 'wb') as outfile:
        writer = UnicodeWriter(outfile)
        writer.writerow(['label', 'clean_text'])
        for label,text in zip(tweet_labels, tweet_texts):
            writer.writerow([label, text])
            sys.stdout.write('-')

    MIN_D = 20
    MAX_D = .5*len(dictionary)

    with open(DICTIONARY_COUNT_FILENAME, 'wb') as outfile:
        writer = UnicodeWriter(outfile)
        writer.writerow(['word', 'count'])
        for word in sorted(dictionary.keys()):
            if dictionary[word] < MIN_D or dictionary[word] > MAX_D:
                continue
            writer.writerow([word, dictionary[word]])
            sys.stdout.write('~')
