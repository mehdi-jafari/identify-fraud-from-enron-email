#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot

# import numpy as np

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL",0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

# tt = np.sort(data, axis=0)
# print tt
# print 'gggg'
# print data

test = []
for key in data_dict:
	if data_dict[key]['salary'] != "NaN" and data_dict[key]['bonus'] != "NaN":
		test.append((key, float(data_dict[key]['salary']) * float(data_dict[key]['bonus'])))

test.sort(key=lambda tup: tup[1])
print test

 	
#print data_dict
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


