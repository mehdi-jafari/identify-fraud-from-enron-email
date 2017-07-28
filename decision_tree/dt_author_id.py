#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import tree
from sklearn.metrics import accuracy_score

print  len(features_train[0])

clf2 = tree.DecisionTreeClassifier(min_samples_split=40)

print '1'
clf2.fit(features_train, labels_train)

print '1'
pred2 = clf2.predict(features_test)

print '1'
#prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())


acc_min_samples_split_2 = accuracy_score(pred2, labels_test)

print acc_min_samples_split_2

#########################################################