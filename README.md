# Identify Fraud from Enron Email by Mehdi Jafari


## Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will build a person of interest identifier based on financial and email data made public as a result of the Enron scandal.

## Data Exploration

* Features:
        The following features are available in the dataset that are devided into two main categories:

    **financial features** :  ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 

    **email features** : ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']

    Selected features are the follwoings:
    
    **selected_financial_features** = ['salary', 'bonus', 'total_payments', 'expenses' ,'deferred_income' , 'total_stock_value', 'restricted_stock']
    **selected_email_features** = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
* General information 
The dataset has 146 data points and 18 datat points are Point of intrests(POI)

    | FeatureName               | Number of NAN values |
    | -------------             | ------------- |
    | salary                    | 51  |
    | to_messages               | 60  |
    | total_payments            | 21  |
    | bonus                     | 64  |
    | total_stock_value         | 20  |
    | expenses                  | 51  |
    | from_messages             | 60  |
    | from_this_person_to_poi   | 60  |
    | poi                       | 0   |
    | deferred_income           | 97  |
    | shared_receipt_with_poi   | 60  |
    | restricted_stock          | 36  |
    | from_poi_to_this_person   | 60  |

# Outlier
I found the follwoing outliers in our dataset; all of them are removed.
* "TOTAL" data point is clearly an outlier
* "LOCKHART EUGENE E" is an entry with all missing financial values
* "THE TRAVEL AGENCY IN THE PARK" is not a person and by checking the endpoint, almost all its values for financial features  are  NAN and all of Email features are missing.

# Adding new features
The following features are added to the dataset:
* from_this_person_to_poi_ratio
* from_poi_to_this_person_ratio
* bonus_ratio