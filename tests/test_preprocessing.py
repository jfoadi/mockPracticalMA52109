###
## Testing preprocessing
## Katherine Hall - University of Bath
## November 25 2026
###

import unittest

import numpy as np
import pandas as pd
from cluster_maker.preprocessing import  select_features, standardise_features

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        #sample dataframe containing two numerical columns and a non-numerical label column
        self.df = pd.DataFrame({
            'feature_1': [1.2, 3.4, 5.6, 7.8, 9.0],
            'feature_2': [100.0, 200.0, 300.0, 400.0, 500.0],
            'group_label': ['Group_A', 'Group_B', 'Group_C', 'Group_D', 'Group_E'],
        })

    #Test 1: testing whether everything is numerical for the KMeans
    def test_select_features_rejects_non_numeric_columns(self):
        
        with self.assertRaises(TypeError):
            select_features(self.df,['feature_1','group_label']) #will fail because group_label doesnt contain numerical values 

    #Test 2: testing whether there are any issues with the column names 
    def test_select_features_raises_key_error_on_missing_column(self): 
        
        with self.assertRaises(KeyError):
            select_features(self.df, ['feature_1','missing_column']) #will fail because 'missing_column' does not exist

    
    #Test 3: testing whether there are any scaling errors
    def test_standardise_features_zero_mean_unit_variance(self):
        
        X = self.df[['feature_1','feature_2']].to_numpy(dtype=float) #converts the two columns into a 2D array
        X_scaled = standardise_features(X) #transforms the numbers 

        self.assertEqual(X_scaled.shape, X.shape) #checks that the output array has the same shape as input array
        self.assertTrue(np.allclose(X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-7)) #checks that the mean of each transformed 
        #column is 0 with a floating point tolerance of 1e-7
        self.assertTrue(np.allclose(X_scaled.std(axis=0), [1.0, 1.0], atol=1e-7)) #checks that the standard deviation of each 
        #transformed column is 1 with floating point tolerance of 1e-7

if __name__ == "__main__":
    unittest.main()