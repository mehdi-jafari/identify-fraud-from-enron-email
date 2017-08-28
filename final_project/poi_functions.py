#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import train_test_split


def scatterPlot(data_dict, features, title):
	data = featureFormat(data_dict, features)

	for datapoint in data:
		x = datapoint[0]
		y = datapoint[1]
		plt.scatter( x, y)
	
	plt.xlabel(features[0])
	plt.ylabel(features[1])
	plt.title(title)
	plt.show()

def getGeneralInfo(data_dict , features_list):	
	numberOfPOI = 0

	NaN_values = {}
	for feature in features_list:
		NaN_values[feature] = 0
			
	for key in data_dict:
	    for feature in data_dict[key]:
	    	if feature == 'poi' and data_dict[key]['poi'] == True:
	    		numberOfPOI+=1

	    	if feature in features_list and data_dict[key][feature] == "NaN":	    		
	    		NaN_values[feature] += 1
	
	print 'There are {} features available in the dataset'.format(len(data_dict[data_dict.keys()[0]].keys()))
	print 'Selected features ({}) are {} '.format(len(features_list), str(features_list).translate(None, "'"))
	print 'The dataset has {}  data points'.format(len(data_dict))
	print 'There are {} POI in the dataset'.format(numberOfPOI)	

	for feature in NaN_values:
		print 'There are {} missing values in {} feature.'.format(str(NaN_values[feature]), feature)

def findDatapointsWithAllNanValues(data_dict , features_list , print_title):	
	
	result = {}

	for key in data_dict:
		result[key] = 0
		for feature in features_list :
			if data_dict[key][feature] == "NaN" :
				result[key] += 1
	filtered_dict = {k for (k,v) in result.items() if v == len(features_list)}

	print print_title
	print filtered_dict
	#print result

def printOutliers(data_dict, featurName, topElements):	
	data = featureFormat(data_dict, [featurName])
	# print type(data)
	# print data
	print '------------------------------- Feature {} ----------------------------------------'.format(featurName)
	

	data[::-1].sort(axis=0)	
	# print data

	topElements =  data[0:topElements]
	
	actualItems = []
	NaNItems = []
	for key in data_dict:
		if data_dict[key][featurName] in topElements:
			actualItems.append((key, float(data_dict[key][featurName])))
		if data_dict[key][featurName] == 'NaN':
			NaNItems.append(key)
	
	print 'Top ' + featurName 
	print actualItems

def createNewFeatures(data_dict, feature_combination):
	for outputFeature, value in feature_combination.items():
		first_feature = value[0]
		second_feature = value[1]		
		for key in data_dict:
			if data_dict[key][first_feature] == "NaN" or data_dict[key][second_feature] == "NaN" or data_dict[key][second_feature] == 0:
				data_dict[key][outputFeature] = "NaN"
			else:
				data_dict[key][outputFeature] = float(data_dict[key][first_feature])/float(data_dict[key][second_feature])


	return data_dict

def find_the_best_k_number_of_features(dataset, labels, features, features_list):
	result_accuracy = {}
	result_precision = {}
	result_recall = {}
	for k in xrange(1,10):
		print '****************************************START naive_bayes With {} features ***************************'.format(k)
		test_features = remove_low_variant_features(labels, features, features_list , k)
		test_features = ['poi'] + test_features
		print test_features
		from sklearn.naive_bayes import GaussianNB
		clf = GaussianNB()
		result = test_classifier(clf, dataset, test_features, folds = 1000)
		print '****************************************END naive_bayes With {} features ***************************'.format(k)
		result_accuracy[k] = result[0]
		result_precision[k] = result[1]
		result_recall[k] = result[2]


	import matplotlib.pylab as plt
	accuracy_lists = sorted(result_accuracy.items()) # sorted by key, return a list of tuples
	x_a, y_a = zip(*accuracy_lists) # unpack a list of pairs into two tuples
	plt.plot(x_a, y_a)

	precision_lists = sorted(result_precision.items()) # sorted by key, return a list of tuples
	x_p, y_p = zip(*precision_lists) # unpack a list of pairs into two tuples
	plt.plot(x_p, y_p)

	recall_lists = sorted(result_recall.items()) # sorted by key, return a list of tuples
	x_r, y_r = zip(*recall_lists) # unpack a list of pairs into two tuples
	plt.plot(x_r, y_r)
	plt.legend(['Accuracy', 'Precision', 'Recall'], loc='upper left')
	plt.show()
	print 'ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg'
	print result_accuracy
	print 'ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg'

		

	# result = {}
	# folds= 1000
	# from sklearn.naive_bayes import GaussianNB
	# from sklearn.cross_validation import StratifiedShuffleSplit
	# print 'hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh'

	# for k in xrange(5,10):
	# 	print k
	# 	# for feature in features_list_org:
	# 	# 	# Replace 'NaN' with 0
	# 	# 	for name in dataset:
	# 	# 		data_point = dataset[name]
	# 	# 	if not data_point[feature]:
	# 	# 		data_point[feature] = 0
	# 	# 	elif data_point[feature] == 'NaN':
	# 	# 		data_point[feature] =0
	# 	feature_list = []
	# 	feature_list = remove_low_variant_features(labels, features, features_list_org , k)
	# 	feature_list = ['poi'] + feature_list
	# 	clf = GaussianNB()

	# 	data = featureFormat(dataset, feature_list, sort_keys = True)
	# 	labels, features = targetFeatureSplit(data)
	# 	cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
	# 	true_negatives = 0
	# 	false_negatives = 0
	# 	true_positives = 0
	# 	false_positives = 0
	# 	for train_idx, test_idx in cv:
	# 		features_train = []
	# 		features_test  = []
	# 		labels_train   = []
	# 		labels_test    = []
	# 		for ii in train_idx:
	# 			features_train.append( features[ii] )
	# 			labels_train.append( labels[ii] )
	# 		for jj in test_idx:
	# 			features_test.append( features[jj] )
	# 			labels_test.append( labels[jj] )
	# 		### fit the classifier using training set, and test on test set
	#         clf.fit(features_train, labels_train)
	#         predictions = clf.predict(features_test)
	#         for prediction, truth in zip(predictions, labels_test):
	#             if prediction == 0 and truth == 0:
	#                 true_negatives += 1
	#             elif prediction == 0 and truth == 1:
	#                 false_negatives += 1
	#             elif prediction == 1 and truth == 0:
	#                 false_positives += 1
	#             elif prediction == 1 and truth == 1:
	#                 true_positives += 1
	#             else:
	#                 print "Warning: Found a predicted label not == 0 or 1."
	#                 print "All predictions should take value 0 or 1."
	#                 print "Evaluating performance for processed predictions:"
	#                 break
	# 	try:

	# 		total_predictions = true_negatives + false_negatives + false_positives + true_positives
	# 		accuracy = 1.0*(true_positives + true_negatives)/total_predictions
	# 		precision = 1.0*true_positives/(true_positives+false_positives)
	# 		recall = 1.0*true_positives/(true_positives+false_negatives)
	# 		f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
	# 		f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
	# 		result[k] = total_predictions
	# 	except:
	# 		print "Got a divide by zero when trying out:", clf
	# 		print "Precision or recall may be undefined due to a lack of true positive predicitons."
	# print result

def remove_low_variant_features(labels, features, features_list , k):

	from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
	threshold = .8 * (1 - .8)
	vt = VarianceThreshold(threshold=threshold)
	features = vt.fit_transform(features)
	
	vt = SelectKBest(f_classif, k=k)
	vt.fit_transform(features, labels)
	
	scores = zip(features_list[1:],vt.scores_)
	sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
	
	print sorted_scores
	selected_features = list(map(lambda x: x[0], sorted_scores))[0:k]	
	
	print selected_features
	return selected_features

def clf_naive_bayes(dataset, features_list):
	print '****************************************START naive_bayes***************************'
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()
	test_classifier(clf, dataset, features_list, folds = 1000)
	print '****************************************END naive_bayes***************************'
	
def clf_decisionTree(dataset, features_list):
	print '****************************************START decisionTree***************************'
	from sklearn import tree
	clf = tree.DecisionTreeClassifier()
	test_classifier(clf, dataset, features_list, folds = 1000)	
	print '****************************************END decisionTree***************************'

def clf_KNeighbors(dataset, features_list):
	print '****************************************START KNeighbors***************************'
	from sklearn.neighbors import KNeighborsClassifier

	clf = KNeighborsClassifier()
	test_classifier(clf, dataset, features_list, folds = 1000)

	print '****************************************END KNeighbors***************************'


def clf_best_params_KNeighbors(dataset, features_list):
	print '****************************************START Tunned KNeighbors***************************'
	
	# based on grid search n_neighbors =1 and  algorithm='ball_tree' gives the best result
	from sklearn.grid_search import GridSearchCV
	from sklearn import neighbors
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn import cross_validation
	from time import time

	startTime	  = time()
	y       	  = np.sign(np.arange(-5.5,14))
	nFolds 		  = 4
	metrics       = ['minkowski','euclidean','manhattan'] 
	weights       = ['uniform','distance']
	numNeighbors  = np.arange(6,10)
	algorithms 	  = ['ball_tree','kd_tree','brute','auto']
	param_grid    = dict(knn__metric=metrics,knn__weights=weights,knn__n_neighbors=numNeighbors, knn__algorithm=algorithms)
	cv           = cross_validation.StratifiedKFold(y,nFolds)
	grid = GridSearchCV(Pipeline([('scale', StandardScaler()), ('knn', neighbors.KNeighborsClassifier())]), param_grid=param_grid, scoring='recall', cv=cv)

	test_classifier(grid, dataset, features_list)
	print 'Tuning of KNN parameters took {} seconds with these parameters {} '.format(round(time()-startTime, 3), grid.best_params_)
	
	print '****************************************END Tunned KNeighbors***************************'

def clf_best_params_DecisionTree(dataset, features_list):
	print '****************************************START Tunned DecisionTree***************************'
	
	# based on grid search n_neighbors =1 and  algorithm='ball_tree' gives the best result
	from sklearn.grid_search import GridSearchCV
	from sklearn import tree
	from time import time
	from sklearn import cross_validation

	startTime	 = time()
	y       	 = np.sign(np.arange(-5.5,14))
	nFolds 		 = 4
	criterions   = ['gini', 'entropy'] 
	splitters    = ['best','random']
	n_estimators = [50, 100, 150, 200]
	max_depths 	 = [2, 4, 6, 8]
	min_splits   = [10, 20, 40]
	param_grid   = dict(max_depth = max_depths, min_samples_split = min_splits)
	cv           = cross_validation.StratifiedKFold(y,nFolds)
	grid 		 = GridSearchCV(tree.DecisionTreeClassifier(),param_grid=param_grid, scoring='recall', cv=cv)

	test_classifier(grid, dataset, features_list)
	print 'Tuning of DecisionTree parameters took {} seconds with these parameters {}'.format(round(time()-startTime, 3), grid.best_params_)
	
	print '****************************************END Tunned DecisionTree***************************'	