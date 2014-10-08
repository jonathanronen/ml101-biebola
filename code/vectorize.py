"""
For Intro to Data Science, NYU 2014.

Reads clean texts and dictionary, makes BOW vectors.

@auth jonathanronen
"""

CLEAN_DATA_FILENAME = 'data/clean_data.csv'
DICTIONARY_COUNT_FILENAME = 'data/dictionary.csv'
TFM_FILENAME = 'data/tfm.m'
LABEL_FILENAME = 'data/y.m'


import sys
import pickle
import scipy as sp
import scipy.io
import scipy.sparse
from smappPy.unicode_csv import UnicodeReader

if __name__ == '__main__':

    texts, labels = list(), list()
    with open(CLEAN_DATA_FILENAME, 'rb') as f:
        reader = UnicodeReader(f)
        header = reader.next()
        for row in reader:
            texts.append(row[1])
            labels.append(row[0])

    dictionary = list()
    with open(DICTIONARY_COUNT_FILENAME, 'rb') as f:
        reader = reader = UnicodeReader(f)
        header = reader.next()
        for row in reader:
            dictionary.append(row[0])

    TFM = [[tweet.count(word) for word in dictionary] for tweet in texts]

    m = sp.sparse.csr_matrix(TFM)
    with open(TFM_FILENAME, 'wb') as f:
        sp.io.mmwrite(f, m)
