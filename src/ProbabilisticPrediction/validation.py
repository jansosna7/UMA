'''
*===========================================================================*
*				Authors: Jan Sosnowski & Marcin Latawiec                    *
*===========================================================================*
'''

#Script for experiments

import os.path
import pandas as pd

'''
Files: 
- breast-cancer.csv
- customer_data.csv
- rice.csv
'''

dirname = os.path.dirname(__file__)
filename_cancer = os.path.join(dirname, '../resources/breast-cancer.csv')
filename_customerdata = os.path.join(dirname, '../resources/customer_data.csv')
filename_rice = os.path.join(dirname, '../resources/rice.csv')

#data handlers
''' breast cancer data '''
#columns_names_cancer =  ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean","radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se","symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst","fractal_dimension_worst"]
#data = pd.read_csv(filename_cancer, skiprows=1, header=None, names=columns_names_cancer)

''' customer_data '''
columns_names_customer_data = ["fea_1", "fea_2", "fea_3", "fea_4", "fea_5", "fea_6", "fea_7", "fea_8", "fea_9", "fea_10", "fea_11"]
data = pd.read_csv(filename_customerdata, skiprows=1, header=None, names=columns_names_customer_data)