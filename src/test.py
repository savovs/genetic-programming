import os
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import accuracy_score, mean_squared_error
from deap import algorithms
from math import sqrt
from operator import itemgetter

from ga import toolbox, stats, hall_of_fame

location = os.path.join(os.getcwd(), os.path.dirname(__file__))
data_path = os.path.join(location, '../UCP_Dataset.csv')

df = pd.read_csv(data_path, sep=';')

print(df.query())


# kf = KFold(n_splits = 10, shuffle = True)


# for train, test in kf.split(df):
# 	print("%s %s" % (train, test))


