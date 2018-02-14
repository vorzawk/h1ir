import json
import re
from nltk.stem.porter import PorterStemmer

# Read the training data file and extract all the features and labels
with open('training_data_tiny.json') as file:
    listFeatures = []
    listLabels = []
    for line in file:
        data = json.loads(line)
        review_text = data['text'].lower()
        listTokens = re.findall('\w+', review_text)
        stemmer = PorterStemmer()
        # dictFeatures contains all the features for the current review, worf
        # counts, votes, stars etc.
        dictFeatures = {}
        for word in listTokens:
            stemToken = stemmer.stem(word)
            if stemToken in dictFeatures.keys():
                dictFeatures[stemToken] += 1
            else:
                dictFeatures[stemToken] = 1
        listLabels.append(data['label'])
        listFeatures.append(dictFeatures)

# print('\n'+'features for each review')
# for featVec in listFeatures:
#     print(featVec)

# There are 40000 reviews and roughly 40000 distinct tokens, so the total size of the
# feature matrix is ~1600000000. Assuming each element is a byte, this requires
# 1.6GB memory! This results in a memory error as expected ...
# The only choice is to use sparse matrices

from sklearn.feature_extraction import DictVectorizer
import numpy as np
# Not setting sparse=False causes sklearn to do the right thing!
vectorizer = DictVectorizer()
X = vectorizer.fit_transform(listFeatures)
# X is a sparse matrix which represents the unlabeled training data

# Convert labels into boolean values (required for computing precision and
# recall)
y = [label == 'Food-relevant' for label in listLabels]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier()
knn1.fit(X_train,y_train)

from sklearn.tree import DecisionTreeClassifier
dt1 = DecisionTreeClassifier()
dt1.fit(X_train,y_train)

from sklearn.naive_bayes import MultinomialNB
nb1 = MultinomialNB()
nb1.fit(X_train,y_train)

from sklearn.svm import SVC
svm1 = SVC()
svm1.fit(X_train,y_train)

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

print('Setting 1:')
print('Accuracy scores')
y_pred = knn1.predict(X_test)
print('KNeighbors:\nAccuracy: {}'.format(accuracy_score(y_test, y_pred)))

y_pred = dt1.predict(X_test)
print('DecisionTree:\nAccuracy: {}'.format(accuracy_score(y_test, y_pred)))

y_pred = nb1.predict(X_test)
print('NaiveBayes:\nAccuracy: {}'.format(accuracy_score(y_test, y_pred)))

y_pred = svm1.predict(X_test)
print('SVM:\nAccuracy: {}'.format(accuracy_score(y_test, y_pred)))

print('Precision and recall for \'Food Relevant\'')
y_pred = knn1.predict(X_test)
print('KNeighbors:\nprecision: {}, recall: {}'.format(precision_score(y_test, y_pred),
                                               recall_score(y_test, y_pred)))

y_pred = dt1.predict(X_test)
print('DecisionTree:\nprecision: {}, recall: {}'.format(precision_score(y_test, y_pred),
                                               recall_score(y_test, y_pred)))

y_pred = nb1.predict(X_test)
print('Naive_bayes:\nprecision: {}, recall: {}'.format(precision_score(y_test, y_pred),
                                               recall_score(y_test, y_pred)))

y_pred = svm1.predict(X_test)
print('SVM:\nprecision: {}, recall: {}'.format(precision_score(y_test, y_pred),
                                               recall_score(y_test, y_pred)))

print('Precision and recall for \'Food irrelevant\'')
# The "positive" class is now flipped, precision and recall require info
# regarding the class label which corresponds to the "relevant" label. So, the
# same classifier works, but the labels have to be flipped.

y_pred = knn1.predict(X_test)
print('initial y_pred : {}'.format(y_pred))
y_pred = [not label for label in y_pred]
print('flipped y_pred : {}'.format(y_pred))
print('KNeighbors:\nprecision: {}, recall: {}'.format(precision_score(y_test, y_pred),
                                               recall_score(y_test, y_pred)))

y_pred = dt1.predict(X_test)
y_pred = [not label for label in y_pred]
print('DecisionTree:\nprecision: {}, recall: {}'.format(precision_score(y_test, y_pred),
                                               recall_score(y_test, y_pred)))

y_pred = nb1.predict(X_test)
y_pred = [not label for label in y_pred]
print('Naive_bayes:\nprecision: {}, recall: {}'.format(precision_score(y_test, y_pred),
                                               recall_score(y_test, y_pred)))

y_pred = svm1.predict(X_test)
y_pred = [not label for label in y_pred]
print('SVM:\nprecision: {}, recall: {}'.format(precision_score(y_test, y_pred),
                                               recall_score(y_test, y_pred)))

print('\nSetting 2(5 fold cross validation):')
from sklearn.model_selection import cross_val_score
knn2 = KNeighborsClassifier()
cv_accuracy = cross_val_score(knn2, X, y, cv=5)
cv_precision = cross_val_score(knn2, X, y, cv=5, scoring='precision')
cv_recall = cross_val_score(knn2, X, y, cv=5, scoring='recall')
print('KNeighbors:\nAccuracy: {}\nprecision: {}, recall: \
      {}'.format(cv_accuracy.mean(),cv_precision.mean(), cv_recall.mean()))
dt2 = DecisionTreeClassifier()
cv_accuracy = cross_val_score(dt2, X, y, cv=5)
cv_precision = cross_val_score(dt2, X, y, cv=5, scoring='precision')
cv_recall = cross_val_score(dt2, X, y, cv=5, scoring='recall')
print('DecisionTree:\nAccuracy: {}\nprecision: {}, recall: \
      {}'.format(cv_accuracy.mean(),cv_precision.mean(), cv_recall.mean()))
nb2 = MultinomialNB()
cv_accuracy = cross_val_score(nb2, X, y, cv=5)
cv_precision = cross_val_score(nb2, X, y, cv=5, scoring='precision')
cv_recall = cross_val_score(nb2, X, y, cv=5, scoring='recall')
print('Naive_bayes:\nAccuracy: {}\nprecision: {}, recall: \
      {}'.format(cv_accuracy.mean(),cv_precision.mean(), cv_recall.mean()))

svm2 = SVC()
cv_accuracy = cross_val_score(svm2, X, y, cv=5)
cv_precision = cross_val_score(svm2, X, y, cv=5, scoring='precision')
cv_recall = cross_val_score(svm2, X, y, cv=5, scoring='recall')
print('SVM:\nAccuracy: {}\nprecision: {}, recall: \
      {}'.format(cv_accuracy.mean(),cv_precision.mean(), cv_recall.mean()))
