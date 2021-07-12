# Principle-Feature-Analysis

## Introduction
https://arxiv.org/abs/2101.12720


## Installation
```
pip install ...
```

## PFA

```Python
pfa(path*, number_sweeps, cluster_size, alpha, min_n_datapoints_a_bin, shuffle_feature_numbers, frac, claculate_mutual_information, basis_log_mutual_information)

```

### Parameters
- **path (String, required):** path to the input CSV file
- **number_sweeps (int, default=1):** Number of sweeps of the PFA. The result of the last sweep is returned. In addition, the return of each sweep are interesected and returned as well.
- **cluster_size (int, default=50):** Number of nodes of a subgraph in the principal_feature_analysis.
- **alpha (float, default=0.01):** Level of significance.
- **min_n_data_points_a_bin (int, default=500):**: The minimum number of data points for each bin in the chi-square test.
- **shuffle_feature_numbers (Bool?, default=False):** If True the number of the features is randomly shuffled.
- **frac (int, default=1):** The fraction of the dataset that is used for the analysis. The set is randomly sampled from the input csv.
- **calculate_mutual_information (Bool?, default=False):** If True the mutual information with features from the PFA with the system state is calculated.
- **basis_log_mutual_information (int. default=2):** Basis of the logarithm used in the calculation of the mutual information.

### Files
- **principal_features_depending_system_state[i].txt:**
Lists the indices (related to the rows of the input csv) of the features that depend on the system state (row 0) where [i] is replaced by the number of sweeps. Each row of this file is a subgraph that could not be divided further where a * separates the features on which the system state depends (before *) and the ones on which the system state does not depend (after *).
- **principal_features_depending_system_state_intersection.txt:**
Analog to the “principal_features_depending_system_state[i].txt”. Due to the intersection the information of subgraphs is missed and there is only one feature a row.
- **principal_features_global_indices[i].txt:**
is the result from the dissection of the graph of all input features before testing for dependence to the system state of the sweep [i]. Each row corresponds to a subgraph that could not have been dissected further where the numbers refer to the features stored in the corresponding row of the input csv.
- **global_indices_and_principal_features_state_dependency[i].csv:**
A csv file where for each sweep [i] the first column is the feature number referring to the row of the input csv file and the second row is the p-value from the chi2 test of the feature with the system state. A p-value of 1.1 means that it was not possible to make at least two bins for corresponding feature due to for a second not at least min_n_datapoints_a_bin where left. Consequently the feature is considered as constant and thus independent of the system state.
 

### Returns
- **pf_from_intersection (list):** A list with content analog to the file principal_features_depending_system_state_intersection.txt.
- **data_frame_feature_mutual_information (pandas.DataFrame, if calculate_mutual_information=True):** A Pandas data frame that contains the mutual information with the feature (index related to the row in the input csv) with the system state (row 0 in the input csv).
