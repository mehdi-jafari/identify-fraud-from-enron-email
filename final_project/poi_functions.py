#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import VarianceThreshold


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

def get_best_features(labels, features, features_list):
	
	from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest

	selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
	features = selector.fit_transform(features)

	print 'univariate features'

	print features

	k = 10
	selector = SelectKBest(f_classif, k=10)
	selector.fit_transform(features, labels)
	print "Best features"
	scores = zip(features_list[1:],selector.scores_)
	sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
	print sorted_scores
	optimized_features_list = ['poi'] + list(map(lambda x: x[0], sorted_scores))[0:k]
	
	print(optimized_features_list)

	return features
