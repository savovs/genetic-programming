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

from ga import toolbox, stats, hall_of_fame

location = os.path.join(os.getcwd(), os.path.dirname(__file__))
data_path = os.path.join(location, '../UCP_Dataset.csv')

df = pd.read_csv(data_path, sep=';')

# Drop irrelevant columns
df.drop(['Project_No', 'DataDonator', 'Real_P20'], 1, inplace=True)

# The GA needs numbers, so we need to
# replace categorical columns with dummie columns with binary numbers for each category
# i.e. column "Sector" will be replaced with "Sector_value1", "Sector_value2"...

# Conveniently, this will also put 0 where there is no value present

df = pd.get_dummies(df, columns=['Sector', 'Language', 'Methodology', 'ApplicationType'])

def fitness(individual, dataRows = [], efforts = []):
	# Transform the tree expression in a callable function
	generatedFunction = toolbox.compile(expr=individual)

	funcResults = []

	for row in dataRows:
		# Unpack each row and use values as args
		funcResults.append(generatedFunction(*row))

	# Get mean squared error between actual and predicted effort
	rmse = sqrt(
		mean_squared_error(
			efforts,
			funcResults
		)
	)

	twoPointPrecision = '%.2f' % rmse
	return float(twoPointPrecision),


kf = KFold(n_splits = 10, shuffle = True)
ga_results = []
rms_errors = []

# Split data for training and testing
for trainingIndices, testingIndices in kf.split(df):
	trainingDf = df.loc[trainingIndices]
	testingDf = df.loc[testingIndices]

	testingValues = testingDf.values.tolist() 

	# Train GA, following global vars are used by fitness function
	dataRows = trainingDf.values.tolist()
	efforts = trainingDf[['Real_Effort_Person_Hours']].values.flatten().tolist()

	toolbox.register('evaluate', fitness, dataRows=dataRows, efforts=efforts)

	pop = toolbox.population(n=10)

	last_generation, logbook = algorithms.eaSimple(
		pop,
		toolbox,

		cxpb = 0.9,
		mutpb = 0.1,
		ngen = 10,
		stats = stats,
		halloffame = hall_of_fame,
		verbose = True
	)

	# rms_errors.append(toolbox.evaluate(hall_of_fame[0]))

	ga_results.append({
		'best': {
			'fitness': toolbox.evaluate(hall_of_fame[0]),
			'primitive': hall_of_fame[0]
		},

		'stats': logbook.select('gen', 'min')
	})

	# Test GA