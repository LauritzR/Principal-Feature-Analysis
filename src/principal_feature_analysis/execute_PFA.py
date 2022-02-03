# Copyright with the authors of the publication "A principal feature analysis"


from .find_relevant_principal_features import find_relevant_principal_features
from .get_mutual_information import get_mutual_information
import time
import pandas as pd
import numpy as np


# paramters for the PFA
# number_output_functions: Number of output features that are to be modeled, i.e. the number of components of the vector-valued output-function. The values are stored in the first number_output_functions rows of the csv-file.
# number_sweeps: Number of sweeps of the PFA. The result of the last sweep is returned.
                # In addition, the return of each sweep are interesected and returned as well.
# cluster_size: number of nodes of a subgraph in the principal_feature_analysis
# alpha=0.01: Level of significance
# min_n_datapoints_a_bin: minimum number of data points for each bin in the chi-square test
# shuffle_feature_numbers: if True the number of the features is randomly shuffled
# frac: the fraction of the dataset that is used for the analysis. The set is randomly sampled from the input csv
# calculate_mutual_information: if True the mutual information with features from the PFA with the components of the output function is calculated
# basis_log_mutual_information: basis of the logarithm used in the calculation of the mutual information




def pfa(path, number_output_functions=1, number_sweeps=1, cluster_size=50, alpha=0.01, min_n_datapoints_a_bin=500, shuffle_feature_numbers=0, frac=1, calculate_mutual_information=0, basis_log_mutual_information=2):
    # pf_ds = principal features related to output functions, pf = all principal features
    start_time=time.time()
    list_pf_ds=[]


    # The csv file's content is an m x n Matrix with m - number components of output-function = number features and n = number of data points
    # where the first number components of output-function rows contain the value of the vector-valued output function for each of the n data points
    # e.g. in case of a one-dimensional output function, the first row can be the label for each data point
    data = pd.read_csv(path, sep=',', header=None)
    for sweep in range(0,number_sweeps):
        print("Sweep number: " + str(sweep+1))
        pf_ds,pf,indices_principal_feature_values=find_relevant_principal_features(data,number_output_functions,cluster_size,alpha,min_n_datapoints_a_bin,shuffle_feature_numbers,frac)
        list_pf_ds.append(pf_ds)
        # Output the principal features related to the output function in a list where the numbers correspod to the rows of the input csv-file
        f = open("principal_features_depending_system_state"+str(sweep)+".txt", "w")
        for i in pf_ds:
            for j in i:
                f.write(str(j) + str(","))
            f.write("\n")
        f.close()
        # Output the principal features in a list where the numbers correspond to the rows of the input csv-file
        f = open("principal_features_global_indices"+str(sweep)+".txt", "w")
        for i in pf:
            for j in i:
                f.write(str(j) + str(","))
            f.write("\n")
        f.close()
        np.savetxt("global_indices_and_principal_features_state_dependency"+str(sweep)+".csv", indices_principal_feature_values,delimiter=",")
    print("Time needed for the PFA in seconds: " + str(time.time()-start_time))


    #Intersect the lists of principal features related to the output function
    #All the features corresponding to the returned subgraphs are considered in each list

    list_principal_features_depending_on_system_state_for_intersection=[]
    for i in list_pf_ds:
        intermediate_list = []
        for j in i:
            for k in j:
                if k !='*':
                    intermediate_list.append(k)
        list_principal_features_depending_on_system_state_for_intersection.append(intermediate_list)
    pf_from_intersection=list_principal_features_depending_on_system_state_for_intersection[0]
    if number_sweeps > 1:
        for i in range(1, len(list_principal_features_depending_on_system_state_for_intersection)):
            pf_from_intersection=list(set(pf_from_intersection).intersection(set(list_principal_features_depending_on_system_state_for_intersection[i])))
        f = open("principal_features_depending_system_state_intersection.txt", "w")
        for i in pf_from_intersection:
            f.write(str(i)+str(","))
        f.close()

    # Outputs a list of DataFrames where the index feature refers to the row in the csv-file.
    # The mutual information is calculated between the feature represented in the first row of the data frame and the feature referred to in the index feature column.
    # The first row is consequently the mutual information of the corresponding component of the output-function with itself
    if calculate_mutual_information==1:
        print("Calculating mutual information")
        list_data_frame_feature_mutual_information=get_mutual_information(data,number_output_functions,pf_from_intersection,min_n_datapoints_a_bin,basis_log_mutual_information)
        for i in range(0,len(list_data_frame_feature_mutual_information)):
            print(list_data_frame_feature_mutual_information[i])
        return pf_from_intersection, list_data_frame_feature_mutual_information
        
    return pf_from_intersection
