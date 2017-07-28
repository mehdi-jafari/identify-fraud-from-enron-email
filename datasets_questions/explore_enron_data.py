#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np
import pandas as pd


enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
df = pd.DataFrame.from_dict(enron_data, orient='index')

#print df.loc['Lay Kenneth l'.upper()]

# print enron_data['Lay Kenneth l'.upper()]
# print enron_data['Skilling Jeffrey k'.upper()]

#print df['Lay Kenneth l'.upper()]
#print len(df.loc[df['salary'] != 'NaN'])
print len(df.loc[(df['poi'] == True)])
#print len(df.loc[(df['total_payments'] == 'NaN')])