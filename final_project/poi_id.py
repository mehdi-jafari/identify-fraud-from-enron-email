#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from poi_functions import showBoxPlot, printOutliers, printGeneralInfo

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_payments'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop( 'TOTAL', 0 )

printGeneralInfo(data_dict, features_list)
### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict
data = featureFormat(my_dataset, features_list)

# showBoxPlot(data, 1, "Salary Box Plot")
# printOutliers(data, data_dict, 1, 5, 5, "salary")

# showBoxPlot(data, 2, "Bonus Box Plot")
# printOutliers(data, data_dict, 2, 5, 5, "bonus")

# showBoxPlot(data, 3, "Total Payments Box Plot")
# printOutliers(data, data_dict, 3, 5, 5, "total_payments")



# showBoxPlot(data, 2, "Bonus Box Plot")
# printOutliers(data, 2, 5, 5, "bonus")



for d in data:
	salary = d[1]
	bonus = d[2]
	plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
# plt.show()

# test = []
# for key in data_dict:
# 	if data_dict[key]['salary'] != "NaN" and data_dict[key]['bonus'] != "NaN":
# 		test.append((key, float(data_dict[key]['salary']) * float(data_dict[key]['bonus'])))

# test.sort(key=lambda tup: tup[1])
# print test


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)