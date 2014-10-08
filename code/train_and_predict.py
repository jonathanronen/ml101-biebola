import numpy as np
import scipy as sp
import scipy.io

from smappPy.unicode_csv import UnicodeReader
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

TFM_FILENAME = 'data/tfm.m'
LABELS_FILENAME = 'data/clean_data.csv'
RAW_DATA= 'data/tweets.csv'

if __name__ == '__main__':
    tfm = sp.sparse.csr_matrix(sp.io.mmread(TFM_FILENAME))
    labels = [row[0] for row in UnicodeReader(open(LABELS_FILENAME,'rb'))][1:]
    Y = [0 if label == 'E' else 1 for label in labels]

    X_train, Y_train = tfm[:10000], Y[:10000]
    X_test, Y_test   = tfm[10000:], Y[10000:]

    clf = MultinomialNB()
    clf.fit(X_train, Y_train)
    yhat = clf.predict(X_test)

    print "confusion matrix"
    print "================\n"
    print confusion_matrix(Y_test, yhat)
    print "\n\n\n"

    raw_texts = [row[-1] for row in UnicodeReader(open(RAW_DATA, 'rb'))][1:]

    print "Some ebola tweets we mistake for bieber:"
    print "=======================================\n"
    ebola_bieber = [i for i,(actual,pred) in enumerate(zip(Y_test, yhat)) if actual == 0 and pred == 1]
    for ebi in ebola_bieber[:15]:
        print raw_texts[10000+ebi]

    print "\n\n\n"
    print "Some bieber tweets we mistake for ebola:"
    print "=======================================\n"
    bieber_ebola = [i for i,(actual,pred) in enumerate(zip(Y_test, yhat)) if actual == 1 and pred == 0]
    for ebi in bieber_ebola[:15]:
        print raw_texts[10000+ebi]

