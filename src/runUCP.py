# Dataset: http://openscience.us/repo/effort/effort-other/ucp.html

import os
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from deap import algorithms, gp
from math import sqrt
from operator import itemgetter
from numpy import mean, std
from matplotlib import pyplot as plt

from ga import toolbox_from_pset, stats, hall_of_fame

location = os.path.join(os.getcwd(), os.path.dirname(__file__))
data_path = os.path.join(location, '../UCP_Dataset.csv')

df = pd.read_csv(data_path, sep=';')

# Drop irrelevant columns
df.drop(['Project_No', 'DataDonator', 'Real_P20'], 1, inplace=True)

# Fill empty values with zeros
df = df.fillna(0)


# The GA needs numbers, so we need to
# replace categorical columns with dummie columns with binary numbers for each category
# i.e. column "Sector" will be replaced with "Sector_value1", "Sector_value2"...

df = pd.get_dummies(df, columns=['Sector', 'Language', 'Methodology', 'ApplicationType'])

# Create primitive set with arity equal to number of columns without effort 
pset = gp.PrimitiveSet('EFFORT', df.shape[1] - 1)
toolbox = toolbox_from_pset(pset)

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
training_best_of_gen_errors = []
testing_errors = []

# Cross-validation training and testing
for trainingIndices, testingIndices in kf.split(df):
	trainingDf = df.loc[trainingIndices]

	# We don't want our algorithm to cheat:
	trainingDfWithoutEffort = trainingDf.drop(['Real_Effort_Person_Hours'], 1)

	testingDf = df.loc[testingIndices]
	testingDfWithoutEffort = testingDf.drop(['Real_Effort_Person_Hours'], 1)


	# Train GA, make fitness function use training data
	toolbox.register(
		'evaluate',
		fitness,
		dataRows=trainingDfWithoutEffort.values.tolist(),
		efforts=trainingDf[['Real_Effort_Person_Hours']].values.flatten().tolist()
	)

	pop = toolbox.population(n=300)

	last_generation, logbook = algorithms.eaSimple(
		pop,
		toolbox,

		cxpb = 0.9,
		mutpb = 0.1,
		ngen = 100,
		stats = stats,
		halloffame = hall_of_fame,
		verbose = True
	)

	training_best_of_gen_errors.append(logbook.select('min'))

	# Test GA, register fitness function with testing data
	toolbox.register(
		'evaluate',
		fitness,
		dataRows=testingDfWithoutEffort.values.tolist(),
		efforts=testingDf[['Real_Effort_Person_Hours']].values.flatten().tolist()
	)

	print('\nBest result:')
	print(hall_of_fame[0])
	print('\n\n')

	fitnessTuple = toolbox.evaluate(hall_of_fame[0])
	testing_errors.append(fitnessTuple[0])

errors_mean = mean(testing_errors)
errors_std = std(testing_errors)
print("Cross-validation mean rmse: %0.2f (+/- %0.2f)" % (errors_mean, errors_std * 2))

# TODO plot
plt.plot(training_best_of_gen_errors, color='gray')


# Weka Mean absolute error baseline
plt.axhline(204.2888, label='baseline', color='red')
plt.axhline(
	errors_mean,
	label="Cross-val mean rmse: %0.2f (+/- %0.2f)" % (errors_mean, errors_std * 2),
	color='green'
)

plt.xlabel('Generation')
plt.ylabel('Root Mean Square Error')
plt.title('UCP dataset')
plt.suptitle('Effort Estimation Using Genetic Programming')
plt.legend()
plt.show()