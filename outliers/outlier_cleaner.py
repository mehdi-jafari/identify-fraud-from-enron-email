#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []    

    ### your code goes here
    for x in xrange(1,90):
        error = net_worths[x] - predictions[x]
        cleaned_data.append((ages[x], net_worths[x], error))
        
    
    cleaned_data.sort(key=lambda tup: tup[2])
    print cleaned_data
    return cleaned_data[10:100]