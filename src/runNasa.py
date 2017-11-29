import os
import pandas as pd
from scipy.io import arff
from deap import algorithms, gp
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from pprint import pprint

from ga import toolbox_from_pset, stats, hall_of_fame 

# Explore data using Weka linear regression to see what variables matter.
'''
=== Weka Cross-validation output ===

Instances:    93
Attributes:   4
              kloc
              effort
              defects
              months
Test mode:    10-fold cross-validation

=== Classifier model (full training set) ===


Linear Regression Model

effort =

     -3.0402 * kloc +
     93.0466 * months +
  -1339.2667

=== Cross-validation ===
=== Summary ===

Correlation coefficient                  0.6804
Root mean squared error                831.2296
'''

# Create primitive set with name and arity
pset = gp.PrimitiveSet('EFFORT', 2)
toolbox = toolbox_from_pset(pset)

location = os.path.join(os.getcwd(), os.path.dirname(__file__))
data_path = os.path.join(location, '../nasa93-dem.arff')

# Parse data
data_blob = arff.loadarff(data_path)
df = pd.DataFrame(data_blob[0])

# Nominal columns contain arbitrary information about categories
# so they won't be useful to determine effort

# Running Weka linear regression only on numeric values shows that
# the 'kloc' and 'months' columns can be used to determine effort
data = df[['kloc', 'months', 'effort']].copy()

kloc = data[['kloc']].values.flatten().tolist()
months = data[['months']].values.flatten().tolist()
effort = data[['effort']].values.flatten().tolist()


def fitness(individual):
	# Transform the tree expression in a callable function
	func = toolbox.compile(expr=individual)

	funcResults = []

	# Evaluate generated function with ARG0 and ARG1
	for loc, month in zip(kloc, months):
		funcResults.append(func(loc, month))

	# Get mean squared error between actual and predicted effort
	rmse = sqrt(
		mean_squared_error(
			effort,
			funcResults
		)
	)

	twoPointPrecision = '%.2f' % rmse
	return float(twoPointPrecision),


toolbox.register('evaluate', fitness)

results = []


# 


# 
for i in range(1):
	pop = toolbox.population(n=300)

	last_generation, logbook = algorithms.eaSimple(
		pop,
		toolbox,

		cxpb=0.9,
		mutpb=0.1,
		ngen=300,
		stats=stats,
		halloffame=hall_of_fame,
		verbose=True
	)

	results.append({
		'best': {
			'fitness': toolbox.evaluate(hall_of_fame[0]),
			'primitive': hall_of_fame[0]
		},

		'stats': logbook.select('gen', 'min')
	})

statNames = ['gen', 'min']

# Plot fittest iteration of algorithm
fittest = min(results, key=lambda item: item['best']['fitness'])

# The best generated function
pprint(str(fittest['best']['primitive']))


# Make a plot figure
fig, ax = plt.subplots()

for i, stat in enumerate(fittest['stats']):
	if i != 0:
		plt.plot(stat, label=statNames[i])

# Weka Mean absolute error baseline
plt.axhline(831.2296, label='baseline', color='red')

plt.xlabel('Generation')
plt.ylabel('Root Mean Square Error')
plt.title('Nasa dataset')
plt.suptitle('Effort Estimation Using Genetic Programming')
plt.legend()

plt.show()