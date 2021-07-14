# Code published under creative commons license CC BY-NC-SA
# Copyright with the authors of the publication "A principal feature analysis"

import numpy as np
import pandas as pd
import math

# Function to calculate the Shannon mutal information of features with one system state
# Input:    data: a csv file in which the first row is the state of the system  and the other rows the values of the features
#           list_variables: sorted list of indices that refer to the features stored in the row of the csv-file
#           min_n_datapoints_a_bin: is supposed to have the same value as min_n_datapoints_a_bin in the function find_relevant_principle_features
#           basis_log_mutual_information:  the basis for the logarithm used to calculate the mutual information.
# Output:   pd_mutual_information: pandas data frame where the first column refers to the indices of the features and the second column refers to the corresponding mutual information of the corresponding feature with the system state (row 0)
#                                  The first entry is the dependence of the system state with itself. The further entries are the dependence of the corresponding row in the csv-file to the system state (first row of csv-file)

def get_mutual_information(data,list_variables,min_n_datapoints_a_bin,basis_log_mutual_information):
    # Calulate the Shannon mutual information
    def make_summand_from_frequencies(x,y):
        if x == 0:
            return 0
        else:
            return x * math.log2( x / y) / math.log2(basis_log_mutual_information)

    list_variables.insert(0,0)

    data_init = data.to_numpy()
    data=data_init[list_variables,:]
    m = data.shape[0]
    n = data.shape[1]
    l = [0 for i in range(0, m)]
    freq_data = [0 for i in range(0, m)]
    left_metrics = [i for i in range(0, m)]
    constant_metrics = []

    for i in range(0, m):
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print("Variable #"f"{i}" " has only constant values")
            left_metrics.remove(i)
            constant_metrics.append(i)
        else:
            list_points_of_support=[]
            datapoints=data[i,:].copy()
            datapoints.sort()
            counter_points=0
            last_complete_bin=0
            for point in range(0,datapoints.size):
                if point>=(datapoints.size-1):
                    list_points_of_support.append(datapoints[datapoints.size-1])
                    break
                counter_points += 1
                if counter_points>=min_n_datapoints_a_bin and datapoints[point]<datapoints[point+1]:
                    list_points_of_support.append(datapoints[point])
                    counter_points=0
                    last_complete_bin=point
            if list_points_of_support[0]>datapoints[0]:
                list_points_of_support.insert(0,datapoints[0])
            list_points_of_support.append(list_points_of_support[-1]+0.1)
            if datapoints[datapoints>=list_points_of_support[-2]].size<min_n_datapoints_a_bin:
                if len(list_points_of_support)>2:
                    list_points_of_support.pop(-2)
            l[i]=list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]
    #Check for constant metrics
    if constant_metrics != []:
        print("List of metrics with constant values:")
        print(constant_metrics)
    if 0 in constant_metrics or len(freq_data[0])<2:
        print("Warning: System state is constant!")

    # Calculate the dependency measure for each feature with the system state
    mutual_info = np.ones((1, len(left_metrics)))
    for i in range(0,1):
        for j in range(0,len(left_metrics)):
            freq_data_product = ((np.histogram2d(data[i,:], data[left_metrics[j],:], bins=(l[i], l[left_metrics[j]]))[0]))/n
            expfreq = (np.outer(freq_data[i], freq_data[left_metrics[j]]))/(n*n)
            mutual_info[0,j]=np.sum(np.array(list(map(make_summand_from_frequencies,freq_data_product.flatten().tolist(),expfreq.flatten().tolist()))))

    pd_mutual_information=pd.DataFrame({"index feature" : list_variables,"mutual information":mutual_info.tolist()[0]})
    pd_mutual_information['index feature']=pd_mutual_information['index feature'].astype(int)
    return pd_mutual_information