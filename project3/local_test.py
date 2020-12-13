import json
from typing import List, Dict

import joblib
import numpy
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


corpus = []
ylist = []
with open('train.json') as f:
    list: List[Dict] = json.load(f)
    for it in list:
        corpus.append(it['data'])
        ylist.append(it['label'])

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,  random_state=1, n_clusters_per_class=1)
#

# vectorizer = TfidfVectorizer()
# vectorizer.fit(corpus)

vectorizer = joblib.load('text_vectorizer')
X = vectorizer.transform(corpus)
y = numpy.array(ylist)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

print("train")  # TODO delete it

# clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,))
# clf.fit(X_train, y_train)
# print("score")  # TODO delete it
# score = clf.score(X_test, y_test)

clf = BaggingClassifier(LinearSVC())
clf.fit(X_train, y_train)
print("score")  # TODO delete it
score = clf.score(X_test, y_test)

# clf = SGDClassifier(loss="hinge", penalty="l2")
# clf.fit(X_train, y_train)
# print("score")  # TODO delete it
# score = clf.score(X_test, y_test)

# clf = KNeighborsClassifier(3)
# clf.fit(X_train, y_train)
# print("score")  # TODO delete it
# score = clf.score(X_test, y_test)

print(score)
