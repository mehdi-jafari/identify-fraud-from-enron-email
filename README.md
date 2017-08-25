# Identify Fraud from Enron Email


## Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will build a person of interest identifier based on financial and email data made public as a result of the Enron scandal.

## Questions
**1- Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]**  

I am investigating the dataser to find the person of intrest who commited the fraud but they are not in the POI list.
The Enron dataset has 146 data points which 18 data points are Point of interests(POI). This dataset contains the following features which are devided into two categrories financial and emial features:

* financial features:  ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

* email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']

The follwoing table displays the number of NAN values in our dataset per feature. As we can see, POI feature has no missing value while some features like loan_advances, director_fees, restricted_stock_deferred, deferral_payments have more than 100 NAN values.

| FeatureName               | Number of NAN values |
    | -------------             | ------------- |
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
* "THE TRAVEL AGENCY IN THE PARK" is not a person and by checking the endpoint, almost all its values for financial features are  NAN and all of Email features are missing.
    
**2- What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]**
 I added the ratio of emails being sent from a peron to POI and vise versa since if the ratio is high then they might be POI as well. Also bonus to total payment is added since if there could be a co-relation between this new feature and POI. The added features are the follwoings:
* from_this_person_to_poi_ratio
* from_poi_to_this_person_ratio
* bonus_ratio

I used SelectKBest algorithm to score the features and select the top 10 features to try calassification

| FeatureName                   | Score               |
    | -------------                 | -------------       |
    | exercised_stock_options       | 24.815079733218194  |
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

 Here are the selected features:
['exercised_stock_options', 'total_stock_value', 'bonus', 'bonus_ratio', 'salary', 'deferred_income', 'long_term_incentive', 'restricted_stock', 'total_payments', 'shared_receipt_with_poi', 'poi']

**3- What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]**
I tried the follwoing algorithms to compare their performances with the low variant features : 

| Classification| Accuracy | Precision | Recall  | F1      |
| ------------- | ---------| --------- | ------- | ------- |
| Naïve Bayes   | 0.83613  | 0.36639   | 0.31400 | 0.33818 |
| Decision Tree | 0.80773  | 0.28376   | 0.29000 | 0.28684 |
| K-Neighbors   |  0.87640 | 0.63878   | 0.16800 | 0.26603 |

As we can see in the table above, Naive bays has the best performance amnong these three algorithms.

**4- What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]**

**5- What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]**

**6- Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]**
