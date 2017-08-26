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

def remove_low_variant_features(labels, features, features_list):

	from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
	threshold = .8 * (1 - .8)
	vt = VarianceThreshold(threshold=threshold)
	features = vt.fit_transform(features)

	#print vt.variances_
	
	k = 10
	vt = SelectKBest(f_classif, k=10)
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
	print '****************************************naive_bayes***************************'
	
def clf_decisionTree(dataset, features_list):
	print '****************************************START decisionTree***************************'
	from sklearn import tree
	clf = tree.DecisionTreeClassifier()
	test_classifier(clf, dataset, features_list, folds = 1000)	
	print '****************************************decisionTree***************************'

def clf_KNeighbors(dataset, features_list):
	print '****************************************START KNeighbors***************************'
	from sklearn.neighbors import KNeighborsClassifier

	clf = KNeighborsClassifier()
	test_classifier(clf, dataset, features_list, folds = 1000)

	print '****************************************KNeighbors***************************'


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