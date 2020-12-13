import argparse
import json
from typing import List, Dict

import numpy
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--training_data', type=str, default='train.json')

args = parser.parse_args()
training_data = args.training_data

corpus = []
ylist = []
with open('train.json') as f:
    list: List[Dict] = json.load(f)
    for it in list:
        corpus.append(it['data'])
        ylist.append(it['label'])

vectorizer = joblib.load('text_vectorizer')
X = vectorizer.transform(corpus)
y = numpy.array(ylist)

clf = BaggingClassifier(LinearSVC())
clf.fit(X, y)

joblib.dump(clf, 'model')
