# Identify Fraud from Enron Email


## Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will build a person of interest identifier based on financial and email data made public as a result of the Enron scandal.

## Questions
**1- Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]**  

I am investigating the dataset to find the person of interest who committed the fraud but they are not in the POI list.
The Enron dataset has 146 data points which 18 data points are Point of interests(POI). This dataset contains the following features which are divided into two categories financial and email features:

* financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

* email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']

The following table displays the number of NAN values in our dataset per feature. As we can see, POI feature has no missing value while some features like loan_advances, director_fees, restricted_stock_deferred, deferral_payments have more than 100 NAN values.


| FeatureName| Number of NAN values |
| ------------- | ---------|
| salary                    | 51  |
| to_messages               | 60  |
| deferral_payments         | 107 |
| total_payments            | 21  |
| exercised_stock_options   | 44  |
| bonus                     | 64  |
| director_fees             | 129 |
| restricted_stock_deferred | 128 |
| from_messages             | 60  |
| total_stock_value         | 20  |
| expenses                  | 51  |
| from_poi_to_this_person   | 60  |
| loan_advances             | 142 |
| email_address             | 35  |
| other                     | 53  |
| from_this_person_to_poi   | 60  |
| poi                       | 0   |
| deferred_income           | 97  |
| shared_receipt_with_poi   | 60  |
| restricted_stock          | 36  |
| long_term_incentive       | 80  |

I found and removed the following outliers from Enron dataset via plotting the data, finding data points with so many missing values or finding endpoint with strange values:

* "TOTAL" data point is clearly an outlier
* "LOCKHART EUGENE E" is an entry with all missing financial values
* "THE TRAVEL AGENCY IN THE PARK" is not a person and by checking the endpoint, almost all its values for financial features are NAN and all of Email features are missing.
    
**2- What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]**
 I added the ratio of emails being sent from a person to POI and vice versa since if the ratio is high then they might be POI as well. Also, bonus to total payment is added since if there could be a co-relation between this new feature and POI. The added features are the followings:
* from_this_person_to_poi_ratio
* from_poi_to_this_person_ratio
* bonus_ratio

I used SelectKBest algorithm to score the features here is the result of scoring:



| FeatureName| Score |
| ------------- | ---------|
| total_stock_value             | 24.182898678566879  |
| bonus                         | 20.792252047181535  |
| bonus_ratio                   | 20.715596247559954  |
| salary                        | 18.289684043404513  |
| deferred_income               | 11.458476579280369  |
| long_term_incentive           | 9.9221860131898225  |
| restricted_stock              | 9.2128106219771002  |
| total_payments                | 8.7727777300916756  |
| shared_receipt_with_poi       | 8.589420731682381   |
| loan_advances                 | 7.1840556582887247  |
| expenses                      | 6.0941733106389453  |
| from_poi_to_this_person       | 5.2434497133749582  |
| other                         | 4.1874775069953749  |
| from_this_person_to_poi       | 2.3826121082276739  |
| director_fees                 | 2.1263278020077054  |
| to_messages                   | 1.6463411294420076  |
| from_this_person_to_poi_ratio | 1.2565738314129471  |
| from_poi_to_this_person_ratio | 0.23029068522964966 |
| deferral_payments             | 0.22461127473600989 |
| from_messages                 | 0.16970094762175533 |
| restricted_stock_deferred     | 0.065499652909942141|

In order to choose the best K value for feature selection, I created the following plot where precision and recall are presented for different values of k. the best result will be with K=6 that has the best possible precision and recall at the same time:

![alt text](https://github.com/mehdi-jafari/identify-fraud-from-enron-email/blob/master/final_project/figure_1.png?raw=true "Accuracy, Precision, and Recall versus number of K-best Features")
 Here are the selected features:
['exercised_stock_options', 'total_stock_value', 'bonus', 'bonus_ratio', 'salary', 'deferred_income', 'poi']
As we can see in the list only one of the three engnineerd features (bonus_ratio) is in the top 10 list.

**3- What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]**
I tried the following algorithms to compare their performances with the low variant features: 

| Classification| Accuracy | Precision | Recall  | F1      |
| ------------- | ---------| --------- | ------- | ------- |
|**Naïve Bayes**| 0.85464  | 0.48876   | 0.38050 | 0.42789 |
| Decision Tree | 0.79907  | 0.30466   | 0.31700 | 0.31071 |
| K-Neighbors   | 0.87657  | 0.68733   | 0.24950 | 0.36610 |

In order to validate that the selected features have better performance than using all features, I tried the same algorithms with all features and here is the result:

| Classification| Accuracy | Precision | Recall  | F1      |
| ------------- | ---------| --------- | ------- | ------- |
| Naïve Bayes   | 0.73900  | 0.22604   | 0.39500 | 0.28753 |
| Decision Tree | 0.79427  | 0.22125   | 0.21550 | 0.21834 |
| K-Neighbors   | 0.87920  | 0.65461   | 0.19900 | 0.30521 |

As we can see in the table above, Naive bays has the best performance among these three algorithms with selected features.

**4- What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]**

Machine learning algorithms require you to set parameters before you use the models. In other words, Machine learning algorithms are parametrized and setting those parameters depend on many factors. The goal is to set those parameters to so optimal values that gives you the best performance in terms of accuracy, precision and recall. Setting those parameters could drive to overfitting which is a common phenomenon in machine learning that we must be aware of to limit and constrain how much detail the model learns once we train the model.

 I used GridSearchCV to tune find the best result for DecisionTree and K- K-Neighbors algorithms by using different parameter. Here is what the best parameters and their performance:

| Classification| time*| Accuracy | Precision | Recall  | F1     | Best parameters|
| ------------- | -------| ---------| --------- | ------- | ------- |-------         |
| Decision Tree | 98| 0.82114  | 0.31250   | 0.21000 | 0.25120 |{'min_samples_split': 10, 'max_depth': 2} |
| K-Neighbors   | 1784|  0.85207 | 0.26174   |0.01950 | 0.03630 |{'knn__algorithm': 'ball_tree', 'knn__weights': 'distance', 'knn__metric': 'manhattan', 'knn__n_neighbors': 9}         |


*time : the time that is taken to find the best parameters.

Since feature scaling does not affect decision tree, it wasn't applied to tune decision tree whereas in tuning KK I used StandardScaler is to preprocess the data.

The algorithm that has the best performance is Naïve Bayes.

**5- What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]**
Validation is the process that we should use to get an assessment of the performance of our classifier or our regression on an independent dataset and helps us to prevent overfitting. A classic mistake of validation is doing validation without shuffling. Since In this project we are dealing with a small and imbalanced dataset I use StratifiedShuffleSplit.


**6- Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]**

I used the following metrics to measure the performance of a classifier algorithm.

* Precision: True Positive / (True Positive + False Positive). Out of all the items labeled as positive, how many truly belong to the positive class which means the proportion of the correct prediction of all the people who are predicted to be poi.
* Recall: is the probability of our algorithm to correctly identify an item provided that the item actually is what that item. in other words, True Positive / (True Positive + False Negative). Out of all the items that are truly positive, how many were correctly classified as positive which means the proportion of the poi the model can detect of all the poi.

## References

http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html

https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
