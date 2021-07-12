# Principle-Feature-Analysis

https://arxiv.org/abs/2101.12720


## Parameters (* required)
- **path (String):** path to the input CSV file
- **number_sweeps (int, default=1):** Number of sweeps of the PFA. The result of the last sweep is returned. In addition, the return of each sweep are interesected and returned as well.
- **cluster_size (int, default=50):** Number of nodes of a subgraph in the principal_feature_analysis.
- **alpha (float, default=0.01):** Level of significance.
- **min_n_data_points_a_bin (int, default=500):**: The minimum number of data points for each bin in the chi-square test.
- **shuffle_feature_numbers (Bool?, default=False):** If True the number of the features is randomly shuffled.
- **frac (int, default=1):** The fraction of the dataset that is used for the analysis. The set is randomly sampled from the input csv.
- **calculate_mutual_information (Bool?, default=False):** If True the mutual information with features from the PFA with the system state is calculated.
- **basis_log_mutual_information (int. default=2):** Basis of the logarithm used in the calculation of the mutual information.
