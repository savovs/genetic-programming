# Dataset: http://openscience.us/repo/effort/effort-other/ucp.html

import os
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from deap import algorithms
from math import sqrt
from operator import itemgetter

# from ga import toolbox, stats, hall_of_fame

location = os.path.join(os.getcwd(), os.path.dirname(__file__))
data_path = os.path.join(location, '../UCP_Dataset.csv')

df = pd.read_csv(data_path, sep=';')

# Drop unusable columns
df.drop(['Project_No', 'DataDonator', 'Real_P20'], 1, inplace=True)
df.dropna(inplace=True)
df = pd.get_dummies(df, columns=['Sector', 'Language', 'Methodology', 'ApplicationType'])

# print(df.head())

# print(pd.factorize( ['Sector', 'Language', 'Methodology', 'ApplicationType'] )[0])


print(df.columns.values)

# kf = KFold(n_splits = 10, shuffle = True)

# for train, test in kf.split(df):
# 	print("%s %s" % (train, test))


