from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import sys

categories = sys.argv[1].split(',')
twenty_train = load_files('20news-bydate-train', categories=categories,
                          shuffle=True, random_state=42, encoding='latin1')

text_clf_pl = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB())])
text_clf = text_clf_pl.fit(twenty_train.data, twenty_train.target)
joblib.dump(text_clf, 'models/' + sys.argv[2] + '.pkl')
