
import os
import pandas as pd
from pprint import pprint
from scipy.io import arff
from deap import algorithms
from math import sqrt
from sklearn.metrics import mean_squared_error
from operator import itemgetter
from matplotlib import pyplot as plt

from ga import toolbox, stats, hall_of_fame 

location = os.path.join(os.getcwd(), os.path.dirname(__file__))
data_path = os.path.join(location, '../kemerer.arff')


# With all columns in, Weka thinks ID matters, but that's not true, 
# because it's just an arbitrary number, so no need to use it.

'''
=== Weka Cross-validation ===

=== Run information ===

Scheme:       weka.classifiers.functions.LinearRegression -S 0 -R 1.0E-8 -num-decimal-places 4
Relation:     kemerer-weka.filters.unsupervised.attribute.Remove-R1
Instances:    15
Attributes:   7
              Language
              Hardware
              Duration
              KSLOC
              AdjFP
              RAWFP
              EffortMM
Test mode:    10-fold cross-validation

=== Classifier model (full training set) ===


Linear Regression Model

EffortMM =

     53.4674 * Hardware +
      0.389  * AdjFP +
   -294.1583

Time taken to build model: 0 seconds

=== Cross-validation ===
=== Summary ===

Correlation coefficient                  0.3664
Mean absolute error                    204.2888
Root mean squared error                281.0625
Relative absolute error                126.9399 %
Root relative squared error            104.7467 %
Total Number of Instances               15

'''



data_blob = arff.loadarff(data_path)
df = pd.DataFrame(data_blob[0])


# Weka shows that these columns corelate the best
data = df[['Hardware', 'AdjFP', 'EffortMM']].copy()

hardwares = data[['Hardware']].values.flatten().tolist()
adjFPs = data[['AdjFP']].values.flatten().tolist()
efforts = data[['EffortMM']].values.flatten().tolist()


def fitness(individual):
	# Transform the tree expression in a callable function
	func = toolbox.compile(expr = individual)

	funcResults = []

	for hardware, adjFP in zip(hardwares, adjFPs):
		funcResults.append(func(hardware, adjFP))

	# Get mean squared error between actual and predicted effort
	rmse = sqrt(
		mean_squared_error(
			efforts,
			funcResults
		)
	)

	twoPointPrecision = '%.2f' % rmse
	return float(twoPointPrecision),


toolbox.register('evaluate', fitness)



results = []

for i in range(1):
	pop = toolbox.population(n = 300)

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

	results.append({
		'best': {
			'fitness': toolbox.evaluate(hall_of_fame[0]),
			'primitive': hall_of_fame[0]
		},

		'stats': logbook.select('gen', 'min')
	})

statNames = ['gen', 'min']

# Plot fittest iteration of algorithm
fittest = min(results, key = lambda item: item['best']['fitness'])

# Make a plot figure
fig, ax = plt.subplots()

# Set plot limits to range of "min" statistic
# ax.set_ylim([min(fittest['stats'][2]), max(fittest['stats'][2])])


for i, stat in enumerate(fittest['stats']):
	if i != 0:
		plt.plot(stat, label = statNames[i])

# Weka Mean absolute error baseline
plt.axhline(204.2888, label = 'baseline', color = 'red')

plt.xlabel('Generation')
plt.ylabel('Root Mean Square Error')
plt.title('Kemerer dataset')
plt.suptitle('Effort Estimation Using Genetic Programming')
plt.legend()
plt.show()