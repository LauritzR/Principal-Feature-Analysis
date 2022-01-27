# Copyright with the authors of the publication "A principal feature analysis"

import numpy as np
import pandas as pd
import math

# Function to calculate the Shannon mutal information of features with the components of the output function
# Input:    data: a csv file in which the first number of components of the output function rows are the values of the components of the output function  and the other rows are the values of the features
#           list_variables: sorted list of indices that refer to the features stored in the row of the csv-file
#           min_n_datapoints_a_bin: is supposed to have the same value as min_n_datapoints_a_bin in the function find_relevant_principle_features
#           basis_log_mutual_information:  the basis for the logarithm used to calculate the mutual information.
# Output:   list_pd_mutual_information: list of pandas data frames where each the first column refers to the indices of the row in the csv (respectively, the variable of which the values are stored there) and the second column refers to the corresponding mutual information of the corresponding feature with the component of the output-function (indicated in the first row of first column)
#                                  The first entry is the mutual information of the corresponding component of the output fuction with itself. The further entries are the mutual information of the corresponding row in the csv-file to the corresponding component of the output function

def get_mutual_information(data,number_output_functions,list_variables,min_n_datapoints_a_bin,basis_log_mutual_information):
    # Calulate the Shannon mutual information
    def make_summand_from_frequencies(x,y):
        if x == 0:
            return 0
        else:
            return x * math.log2( x / y) / math.log2(basis_log_mutual_information)

    # Insert the the indices of the rows where the components of the output functions are stored
    for i in range(0,number_output_functions):
        list_variables.insert(i,i)

    data_init = data.to_numpy()
    data=data_init[list_variables,:]
    m = data.shape[0]
    n = data.shape[1]
    l = [0 for i in range(0, m)]
    freq_data = [0 for i in range(0, m)]
    left_features = [i for i in range(0, m)]
    constant_features = []

    for i in range(0, m): 
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print("Feature #"f"{list_variables[i]}" " has only constant values")
            left_features.remove(i)
            constant_features.append(list_variables[i])
        else:
            # start the binning by sorting the data points
            list_points_of_support = []
            datapoints = data[i, :].copy()
            datapoints.sort()
            last_index = 0
            # go through the data points and bin them
            for point in range(0, datapoints.size):
                if point >= (datapoints.size - 1):  # if end of the data points leave the for-loop
                    break
                # close a bin if there are at least min_n_datapoints_a_bin and the next value is bigger
                if datapoints[last_index:point + 1].size >= min_n_datapoints_a_bin and datapoints[point] < datapoints[point + 1]:
                    list_points_of_support.append(datapoints[point + 1])
                    last_index = point + 1
            if len(list_points_of_support) > 0: # test that there is at least one point of support (it can be if there are only constant value up to the first ones which are less than min_n_datapoints_a_bin
                if list_points_of_support[0] > datapoints[0]: # add the first value as a point of support if it does not exist (less than min_n_datapoints_a_bin at the beginning)
                    list_points_of_support.insert(0, datapoints[0])
            else:
                list_points_of_support.append(datapoints[0])
            list_points_of_support.append(datapoints[-1] + 0.1) # Add last point of support such that last data point is included (half open interals in Python!)
            if datapoints[datapoints >= list_points_of_support[-2]].size < min_n_datapoints_a_bin: # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
                if len(list_points_of_support) > 2:     # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]


    #Check for constant features
    if constant_features != []:
        print("List of features with constant values:")
        print(constant_features)
    for id_output in range(0,number_output_functions):
        if id_output in constant_features or len(freq_data[id_output]) < 2:  # Warn if the output function is constant e.g. due to an unsuitable binning
            print("Warning: Output function " + str(id_output) +  " is constant!")

    # Calculate the mutual information for each feature with the corresponding component of the output function
    list_of_data_frames=[]
    mutual_info = np.ones((1, len(left_features) - number_output_functions + 1))  # number of featuers plus one component of the output-function
    for i in range(0,number_output_functions):
        list_of_features=list(range(number_output_functions,len(left_features)))
        list_of_features.insert(0,i)
        id_features=np.array(list_variables)[left_features]
        
        for j in list_of_features:
            freq_data_product = ((np.histogram2d(data[i,:], data[left_features[j], :], bins=(l[i], l[left_features[j]]))[0])) / n
            expfreq = (np.outer(freq_data[i], freq_data[left_features[j]])) / (n * n)
            if j<number_output_functions:
                mutual_info[0, 0] = np.sum(np.array(list(map(make_summand_from_frequencies, freq_data_product.flatten().tolist(),expfreq.flatten().tolist()))))
            else:
                mutual_info[0,j-number_output_functions+1]=np.sum(np.array(list(map(make_summand_from_frequencies,freq_data_product.flatten().tolist(),expfreq.flatten().tolist()))))
        pd_mutual_information = pd.DataFrame({"index feature": id_features, "mutual information": mutual_info.tolist()[0]})
        pd_mutual_information['index feature'] = pd_mutual_information['index feature'].astype(int)
        list_of_data_frames.append(pd_mutual_information)

    return list_of_data_frames
