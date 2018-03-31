import json
import re
from nltk.stem.porter import PorterStemmer

# Read the training data file and extract all the features and labels
with open('training_data_med.json') as file:
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
 #       dictFeatures['usefulness'] = data['votes']['useful']
#      dictFeatures['stars'] = data['stars']
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

from sklearn.naive_bayes import MultinomialNB
nb1 = MultinomialNB()
nb1.fit(X_train,y_train)

from sklearn.metrics import accuracy_score

y_pred = nb1.predict(X_test)
print('NaiveBayes:\nAccuracy: {}'.format(accuracy_score(y_test, y_pred)))
