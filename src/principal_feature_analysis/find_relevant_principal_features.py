# Copyright with the authors of the publication "A principal feature analysis"

import numpy as np
import scipy.stats
import pandas as pd
from .principal_feature_analysis import principal_feature_analysis

def find_relevant_principal_features(data,number_output_functions,cluster_size,alpha,min_n_datapoints_a_bin,shuffle_feature_numbers,frac):
    # In this function the binning is done, the graph is dissected and the relevant features/variables are detected
    if frac<1: # if frac<1 the feature analysis is done only the fraction of the data, randomly sampled
        data=data.sample(frac=frac,axis='columns',replace=False)
    data=data.to_numpy()
    m = data.shape[0] #number features
    n = data.shape[1] #number of data points
    l = [0 for i in range(0, m)]  # list of lists with the points of support for the binning
    freq_data = [0 for i in range(0, m)] # list of histograms
    left_features = [i for i in range(0, m)]  # list of features that is step by step reduced to the relevant ones
    constant_features = []

    # remove constant features and binning (discretizing the continuous values of our features)
    for i in range(0, m): 
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print("Feature #"f"{i}" " has only constant values")
            left_features.remove(i)
            constant_features.append(i)
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
    print("Binning done!")
    print("List of features with constant values:")
    print(constant_features)
    for id_output in range(0,number_output_functions):
        if id_output in constant_features or len(freq_data[id_output]) < 2:  # Warn if the output function is constant e.g. due to an unsuitable binning
            print("Warning: System state " + str(id_output) +  " is constant!")

    print("Starting principal feature analysis!")
    # list_principal_features = list of resulting principal features
    # smaller_than5 = percent of chi-square tests with bins less than 5 data points
    # smaller_than1 = percent of chi-square tests with bins less than 1 data point
    list_principal_features, smaller_than5, smaller_than1 = principal_feature_analysis(cluster_size, data, number_output_functions, freq_data, l, left_features, alpha, shuffle_feature_numbers)

    # Assign global index to each principal feature
    list_principal_features_global_indices = []
    for i in list_principal_features:
        intermediate_list = []
        for j in i:
            shift_variable = j + number_output_functions  # Output functions, which are not considered in the PFA, are the values 0,1,...,number_output_functions-1 in left_features
            intermediate_list.append(left_features[shift_variable])
        list_principal_features_global_indices.append(intermediate_list)

    print('principal features:')
    print(list_principal_features_global_indices)



    # identify principal features related to the output function again using chi-square
    print("Start calculating dependence on system state")
    principal_features_depending_on_system_state=[]
    principal_features_not_depending_on_system_state = []
    counter_bins_less_than5_relevant_principal_features=0 # number of chi-square tests with less than 5 datapoints a bin
    counter_bins_less_than1_relevant_principal_features=0 # number of chi-square tests with less than 1 datapoint a bin
    counter_number_chi_square_tests_relevant_principal_features=0 # number of total chi-square tests
    indices_principal_feature_values=np.zeros((1, 2))

    for i in list_principal_features_global_indices: # go through any complete graph
        intermediate_list_depending_on_system_state=[]
        intermediate_list_not_depending_on_system_state=[]
        # for each feature within the current complete graph perform a chi-square test
        for j in i:
            if len(freq_data[j]) > 1:
                dependent = 0 # Flag for the input feature j if there is a relation to one output-function
                for id_output in range(0,number_output_functions):
                    counter_number_chi_square_tests_relevant_principal_features +=1
                    freq_data_product = np.histogram2d(data[id_output, :], data[j, :],
                                                bins=(l[id_output], l[j]))[0]
                    expfreq = np.outer(freq_data[id_output], freq_data[j]) / n
                    if sum(expfreq.flatten() < 5) > 0:
                        counter_bins_less_than5_relevant_principal_features += 1
                    if sum(expfreq.flatten() < 1) > 0:
                        counter_bins_less_than1_relevant_principal_features += 1
                    pv = scipy.stats.chisquare(freq_data_product.flatten(), expfreq.flatten(),ddof=-1)[1]
                    # ddof=-1 to have the degrees of freedom of the chi square eaual the number of bins, see corresponding paper (Appendix) for details
                    # if p-value pv is less than alpha the hypothesis that j is independent of the output function is rejected
                    if pv <= alpha:
                        dependent=1
                        break
                if dependent==1:
                    intermediate_list_depending_on_system_state.append(j)
                else:
                    intermediate_list_not_depending_on_system_state.append(j)
            else:
                intermediate_list_not_depending_on_system_state.append(j)
                pv=1.1
            indices_principal_feature_values=np.concatenate((indices_principal_feature_values, np.array([j, pv]).reshape((1, 2))), axis=0)

        #Output the result
        intermediate_list_depending_on_system_state_empty = 1
        intermediate_list_not_depending_on_system_state_empty=1
        if len(intermediate_list_depending_on_system_state)>0:
            intermediate_list_depending_on_system_state_empty=0
        if len(intermediate_list_not_depending_on_system_state)>0:
            intermediate_list_not_depending_on_system_state_empty=0
            principal_features_not_depending_on_system_state.append(intermediate_list_not_depending_on_system_state)

        if intermediate_list_depending_on_system_state_empty==0:
            if intermediate_list_depending_on_system_state_empty==intermediate_list_not_depending_on_system_state_empty:
                print("Ambiguous complete subgraph:")
                print("List with features depending on system state:")
                print(intermediate_list_depending_on_system_state)
                print("List with features not depending on system state:")
                print(intermediate_list_not_depending_on_system_state)
                # if a subgraph contains a node not independent of the output function and a node independent of the node, the classes of nodes are separated by *
                intermediate_list_depending_on_system_state = intermediate_list_depending_on_system_state + ['*'] + intermediate_list_not_depending_on_system_state
            principal_features_depending_on_system_state.append(intermediate_list_depending_on_system_state)
    print("principal features depending on system state:")
    print(principal_features_depending_on_system_state)
    print(str(smaller_than5) + "% of the chi-square test for the adjacency matrix have been performed with a bin less than 5 data points!")
    print(str(smaller_than1) + "% of the chi-square test for the adjacency matrix have been performed with a bin less than 1 data points!")
    print(str(counter_bins_less_than5_relevant_principal_features / counter_number_chi_square_tests_relevant_principal_features * 100) + "% of the chi-square test for finding the relevant principal features have been performed with a bin less than 5 data points!")
    print(str(counter_bins_less_than1_relevant_principal_features / counter_number_chi_square_tests_relevant_principal_features * 100) + "% of the chi-square test for finding the relevant principal features have been performed with a bin less than 1 data points!")
    return principal_features_depending_on_system_state, list_principal_features_global_indices, indices_principal_feature_values
