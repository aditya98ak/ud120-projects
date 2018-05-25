#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
from sklearn.naive_bayes import GaussianNB

# Classifier
clf = GaussianNB()
print('time before fitting up the model is ')
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

# Predicting
print('Time before prediciting...')
t1 = time()
values = clf.predict(features_test)
print('Predicting Time:', round(time()-t1, 3), 's')
print('result is ', values)


#########################################################
### your code goes here ###


#########################################################
