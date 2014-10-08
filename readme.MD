## Example code for tweet classification


#### Code
Code for preprocessing, vectorizing, and classifying tweets.

#### Data
Some tweets about ebola or justin bieber, collected in a 24 hour period 10/6-7 2014.

### Requirements
* `smappPy` [here](https://github.com/SMAPPNYU/smappPy)
* `nltk` use pip
* `numpy`, `scipy` use pip
* `scikit-learn` use pip

### Run
1. The script `preprocessing.py` reads the data from `tweets.csv` and does some word cleanup:
    * lowercase
    * removes punctuation
    * removes words which contain **ebola** or **bieber**
    * removes numbers

2. The script `vectorizer.py` reads the output from `preprocessing.py`, which is `clean_data.csv` and `dictionary.csv`. It then counts word occurences in clean tweet texts, and produces a document-term-matrix. Saves that ti `tfm.m`.

3. The script `train_and_predict.py` uses `sklearn`'s Naive Bayes classifier to train on a training set, and report some results on a held out test set.

* * *

&copy; Jonathan Ronen, New York University 2014