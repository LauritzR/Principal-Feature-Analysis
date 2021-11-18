# Examples

Here you can find three csv-files providing examples to get your hands on the execution of the principal-feature-analysis. The data has been generated with the test_csv_generator_1dim.py and test_csv_generator_2dim.py scripts.

The first step is always the import. You can simply do that via
```Python
from principal_feature_analysis import pfa
```
This imports the main pfa function to use in your code.
It is always required to set the path to the csv-file as a parameter. Depending on the properties of the data it might be necessary to set additional parameters, which will be demonstrated in the following.


## Example 1: test_PFA_1dim_cont_y.csv
The features in this file are continuous and the output function is 1-dimensional. This means we do not have to adjust any hyperparameters as the standard values already cover this case. The code for execution looks the following:
```Python
pfa("test_PFA_1dim_cont_y.csv")
```

## Example 2: test_PFA_1dim_disc_y.csv
Now we have discrete data and an output function which is still 1-dimensional. This does not change anything, the execution still looks like before:
```Python
pfa("test_PFA_1dim_disc_y.csv")
```

## Example 3: test_PFA_2dim_cont_y.csv
The output function in this data is 2-dimensional with continuous features, so we have to adjust the **number_output_functions** parameter.
```Python
pfa("test_PFA_2dim_cont_y.csv", number_output_functions=2)
```

## Optional

The output of the pfa can also be asigned to a variable via
```Python
results = pfa(...)
```
The rest is trying different hyperparameter settings and seeing the effect it has on the results.
