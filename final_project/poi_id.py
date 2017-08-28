#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from poi_functions import getGeneralInfo, findDatapointsWithAllNanValues, scatterPlot, printOutliers, \
createNewFeatures, remove_low_variant_features, clf_naive_bayes, clf_decisionTree, clf_KNeighbors,\
 clf_best_params_KNeighbors, clf_best_params_DecisionTree

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

all_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']

all_features_test = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income','total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list = ['poi'] + financial_features + email_features # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

getGeneralInfo(data_dict, all_features)

#findDatapointsWithAllNanValues(data_dict, financial_features, "List of people with all missing financial feature")
#print data_dict['LOCKHART EUGENE E']

#findDatapointsWithAllNanValues(data_dict, email_features, "List of people with all missing Email feature")
#print data_dict["THE TRAVEL AGENCY IN THE PARK"]

### Task 2: Remove outliers
# "TOTAL" data point is clearly an outlier
data_dict.pop( 'TOTAL', 0 )
# LOCKHART EUGENE E is with all missing financial values
data_dict.pop( 'LOCKHART EUGENE E', 0 )
# it's not a personand by checking the endpoint, it has almost all missing financial features and  all Email features missing  
data_dict.pop( 'THE TRAVEL AGENCY IN THE PARK', 0 )

#Finding outliers from plotting the distrbution

# as we can see in the following diagrams there might be few outliers
#scatterPlot(data_dict, ['total_payments', 'total_stock_value'] , " total_payments and total_stock_value")

# we can get the datapoint that has the biggest total payment and see if we should remove it from the dataset
#printOutliers(data_dict,"total_payments",1)
#printOutliers(data_dict,"total_stock_value",1)
#data_dict.pop( 'LAY KENNETH L', 0 )

scatterPlot(data_dict, ['total_payments', 'total_stock_value'] , " total_payments and total_stock_value")
scatterPlot(data_dict, ['from_poi_to_this_person', 'from_this_person_to_poi'] , " from_poi_to_this_person and from_this_person_to_poi")
scatterPlot(data_dict, ['salary', 'bonus'] ," salary and bonus" )

# end Finding with plotting



### Task 3: Create new feature(s)

new_feature_combinations = {'from_this_person_to_poi_ratio': ['from_messages', 'from_this_person_to_poi'] ,
							'from_poi_to_this_person_ratio': ['to_messages', 'from_poi_to_this_person'] ,
							'bonus_ratio': ['bonus', 'total_payments']
							}

features_list = features_list + ['from_this_person_to_poi_ratio', 'from_poi_to_this_person_ratio', 'bonus_ratio']

my_dataset = createNewFeatures(data_dict, new_feature_combinations)

### Store to my_dataset for easy export below.

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


selected_featureList = ['poi'] + remove_low_variant_features(labels, features, features_list)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf_naive_bayes(my_dataset, selected_featureList)
clf_naive_bayes(my_dataset, all_features_test)

clf_decisionTree(my_dataset, selected_featureList)
clf_decisionTree(my_dataset, all_features_test)

clf_KNeighbors(my_dataset, selected_featureList)
clf_KNeighbors(my_dataset, all_features_test)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# clf_best_params_KNeighbors(my_dataset, features_list)
clf_best_params_KNeighbors(my_dataset, selected_featureList)
clf_best_params_DecisionTree(my_dataset, selected_featureList)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)