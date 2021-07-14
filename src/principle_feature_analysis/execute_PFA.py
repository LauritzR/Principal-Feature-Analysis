# Code published under creative commons license CC BY-NC-SA
# Copyright with the authors of the publication "A principal feature analysis"
from .find_relevant_principal_features import find_relevant_principal_features
from .get_mutual_information import get_mutual_information
import time
import pandas as pd
import numpy as np

# paramters for the PFA
path="/Users/i534747/PycharmProjects/Midgard/Code Paper/all_cases_labeled_train.csv"       # path to the input CSV file
number_sweeps=1 # Number of sweeps of the PFA. The result of the last sweep is returned.
                # In addition, the return of each sweep are interesected and returned as well.
cluster_size=50 # number of nodes of a subgraph in the principal_feature_analysis
alpha=0.01 # alpha=0.01: Level of significance
min_n_datapoints_a_bin=500 # minimum number of data points for each bin in the chi-square test
shuffle_feature_numbers=0 # if 1 the number of the features is randomly shuffled
frac=1 # the fraction of the dataset that is used for the analysis. The set is randomly sampled from the input csv
claculate_mutual_information=0 # if 1 the mutual information with features from the PFA with the system state is calculated
basis_log_mutual_information=2 # basis of the logarithm used in the calculation of the mutual information
def pfa(path, number_sweeps=1, cluster_size=50, alpha=0.01, min_n_datapoints_a_bin=500, shuffle_feature_numbers=0, frac=1, claculate_mutual_information=0, basis_log_mutual_information=2):
    

    # The csv file's content is an m x n Matrix with m - 1 = number features and n = number of data points
    # where the first row contains the value of the output function for each of the n data points
    # e.g. the first row can be the label for each data point
    data = pd.read_csv(path, sep=',', header=None)
    # pf_ds = principal features depending on system state, pf = all principal features
    start_time=time.time()
    list_pf_ds=[]
    
    for sweep in range(0,number_sweeps):
        print("Sweep number: " + str(sweep+1))
        pf_ds,pf,indices_principal_feature_values=find_relevant_principal_features(data,cluster_size,alpha,min_n_datapoints_a_bin,shuffle_feature_numbers,frac)
        list_pf_ds.append(pf_ds)
        # Output the principal features depending on the system state in a list where the numbers correspod to the rows of the input csv-file
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


    #Intersect the lists of principal features depending on the system state
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

    if claculate_mutual_information==1:
        print("Calculating mutual information")
        data_frame_feature_mutual_information=get_mutual_information(data,pf_from_intersection,min_n_datapoints_a_bin,basis_log_mutual_information)
        print(data_frame_feature_mutual_information)
    
    return pf_from_intersection, data_frame_feature_mutual_information