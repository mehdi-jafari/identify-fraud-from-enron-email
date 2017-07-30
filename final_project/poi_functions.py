#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


def removeOutliers():
	print 'gg'


def showBoxPlot(data, featurId, title):
	plt.boxplot( data[:,featurId] )
	plt.title(title)
	plt.show()

def printGeneralInfo(data_dict , features_list):	
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
	
	print ' the dataset has  ' + str(len(data_dict)) + ' endpoints '
	print ' number of Person of Intrest in the dataset is ' + str(numberOfPOI)
	print  ' selected features ( '+ str(len(features_list)) +' ) are the following : ' + str(features_list).translate(None, "'")
	

	print ' there are ' + str(len(data_dict[data_dict.keys()[0]].keys())) +' features available in the dataset'

	for feature in NaN_values:
		print "number of NAN for " + feature  + " : "  + str(NaN_values[feature])


def showScatterPlot(data, featurId, title):
	plt.boxplot( data[:,featurId] )
	plt.title(title)
	plt.show()


def printOutliers(data, data_dict, featurId, topElements , bottomElements, featurName):
	
	print '------------------------------- Feature ' + featurName + '----------------------------------------'

	test = data[:,featurId]
	test[::-1].sort()	
	topElements =  test[0:topElements]	
	bottomElements = test[len(test)-bottomElements-1 : len(test)-1]
	# print test
	topElements = np.unique(topElements)
	bottomElements = np.unique(bottomElements)

	# print type(bottomElements)
	# print bottomElements

	actualItems = []
	NaNItems = []
	for key in data_dict:
		if data_dict[key][featurName] in topElements:
			actualItems.append((key, float(data_dict[key][featurName])))
		if data_dict[key][featurName] == 'NaN':
			NaNItems.append(key)
	
	print 'Top ' + featurName 
	print actualItems

	actualItems = []

	for key in data_dict:
		# print data_dict[key][featurName]
		if data_dict[key][featurName] in bottomElements:
			actualItems.append((key, float(data_dict[key][featurName])))

	print 'Bottom ' + featurName
	print actualItems

	print  featurName + ' feature has '+ str(len(NaNItems)) + ' NaN values'
	print '-----------------------------------------------------------------------' 