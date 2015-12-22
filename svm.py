import numpy as np
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import sys

categories = sys.argv[1].split(',')
twenty_train = load_files('20news-bydate-train', categories=categories,
                          shuffle=True, random_state=42, encoding='latin1')

text_clf_pl = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                              alpha=1e-3, n_iter=5,
                                              random_state=42))])
text_clf = text_clf_pl.fit(twenty_train.data, twenty_train.target)

twenty_test = load_files('20news-bydate-test', categories=categories,
                         shuffle=True, random_state=42, encoding='latin1')
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
mean = np.mean(predicted == twenty_test.target)

print('Accuracy: ' + str(mean))
print(metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names))
