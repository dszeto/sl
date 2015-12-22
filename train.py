import numpy as np
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = load_files('20news-bydate-train', categories=categories, shuffle=True, random_state=42, encoding='latin1')

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

tf_transformer = TfidfTransformer(use_idf=False)
X_train_tf = tf_transformer.fit_transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

text_clf_pl = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf_pl.fit(twenty_train.data, twenty_train.target)

twenty_test = load_files('20news-bydate-test', categories=categories, shuffle=True, random_state=42, encoding='latin1')
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
mean_nb = np.mean(predicted == twenty_test.target)

text_clf_svm_pl = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42)),
])
text_clf_svm = text_clf_svm_pl.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(docs_test)
mean_svm = np.mean(predicted_svm == twenty_test.target)

print('Accuracy  (NB): ' + str(mean_nb))
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print('Accuracy (SVM): ' + str(mean_svm))
print(metrics.classification_report(twenty_test.target, predicted_svm, target_names=twenty_test.target_names))
