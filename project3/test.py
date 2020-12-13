import argparse
import json
from typing import List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--test_data', type=str, default='testdataexample')
parser.add_argument('-m', '--model_file', type=str, default='model_default')

args = parser.parse_args()
test_data = args.test_data
model_file = args.model_file

corpus: List[str] = []
with open(test_data) as f:
    corpus = json.load(f)

vectorizer = joblib.load('text_vectorizer')
X = vectorizer.transform(corpus)

clf = joblib.load(model_file)

for i in range(X.shape[0]):
    print(clf.predict(X[i])[0])
