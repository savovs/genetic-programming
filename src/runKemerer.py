
import os
import pandas as pd
from pprint import pprint
from scipy.io import arff
from deap import algorithms, gp
from math import sqrt
from sklearn.metrics import mean_squared_error
from operator import itemgetter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from ga import toolbox_from_pset, stats, hall_of_fame 

# Explore data using Weka linear regression to see what variables matter.

# With all columns in, Weka thinks ID matters, but that's not true, 
# because it's just an arbitrary number, so no need to use it.
'''
=== Weka Cross-validation ===
Attributes:   7
              Language
              Hardware
              Duration
              KSLOC
              AdjFP
              RAWFP
              EffortMM
Test mode:    10-fold cross-validation

Linear Regression Model

EffortMM = 53.4674 * Hardware +	0.389  * AdjFP + -294.1583 

=== Summary ===

Correlation coefficient                  0.3664
Root mean squared error                281.0625 <---------- Baseline
'''

# Create primitive set with name and arity
pset = gp.PrimitiveSet('EFFORT', 2)
toolbox = toolbox_from_pset(pset)

location = os.path.join(os.getcwd(), os.path.dirname(__file__))
data_path = os.path.join(location, '../kemerer.arff')
data_blob = arff.loadarff(data_path)
df = pd.DataFrame(data_blob[0])


# Weka shows that these columns corelate the best
data = df[['Hardware', 'AdjFP', 'EffortMM']].copy()

hardwares = data[['Hardware']].values.flatten().tolist()
adjFPs = data[['AdjFP']].values.flatten().tolist()
efforts = data[['EffortMM']].values.flatten().tolist()


def fitness(individual):
	# Transform the tree expression in a callable function
	func = toolbox.compile(expr=individual)

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
	pop = toolbox.population(n=300)

	last_generation, logbook = algorithms.eaSimple(
		pop,
		toolbox,

		cxpb = 0.9,
		mutpb = 0.1,
		ngen = 300,
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
fittest = min(results, key=lambda item: item['best']['fitness'])

# The best generated function
pprint(str(fittest['best']['primitive']))


# Make a plot figure
fig, ax = plt.subplots()

# Set plot limits to range of "min" statistic
# ax.set_ylim([min(fittest['stats'][2]), max(fittest['stats'][2])])


for i, stat in enumerate(fittest['stats']):
	if i != 0:
		plt.plot(stat, label=statNames[i])

# Weka Mean absolute error baseline
plt.axhline(281.0625, label='Linear Regression Baseline', color='red')

plt.xlabel('Generation')
plt.ylabel('Root Mean Square Error')
plt.title('Kemerer dataset')
plt.suptitle('Effort Estimation Using Genetic Programming')
plt.legend()
plt.show()

# Example Resulting function
# mul(sqrt(add(mul(sqrt(add(add(sqrt(7.667999712141599), add(sqrt(ARG1), 
# mul(sqrt(div(sqrt(add(ARG0, ARG1)), ARG1)), add(div(ARG1, 
# add(sqrt(2.0194298119934584), div(sqrt(ARG1), sub(9.69260900025867, 
# add(sqrt(sqrt(ARG0)), sqrt(ARG1)))))), add(ARG0, ARG0))))), 
# mul(sqrt(add(ARG1, sqrt(sqrt(ARG0)))), ARG1))), div(ARG1, add(ARG0, 
# div(sqrt(ARG1), sub(9.69260900025867, mul(ARG0, 
# add(sqrt(sqrt(add(sqrt(ARG1), add(mul(ARG0, mul(ARG0, 7.667999712141599)), 
# ARG1)))), ARG0))))))), add(7.667999712141599, 
# mul(sqrt(add(mul(sqrt(sqrt(add(sqrt(sqrt(sqrt(ARG0))), ARG0))), 
# sub(div(ARG1, add(sqrt(sqrt(7.667999712141599)), div(sqrt(ARG1), 
# sub(9.69260900025867, add(sqrt(sqrt(ARG0)), mul(ARG0, mul(ARG0, 
# 7.667999712141599))))))), sub(add(ARG1, sqrt(2.0194298119934584)), 
# add(sqrt(ARG1), div(div(mul(7.667999712141599, 7.667999712141599), 
# sub(div(div(ARG1, 9.012357525628882), sqrt(ARG1)), add(sqrt(ARG0), 
# sqrt(add(ARG0, 5.458368815782998))))), sqrt(sqrt(ARG0))))))), 
# add(add(add(7.667999712141599, 8.421383262353041), ARG0), add(sqrt(mul(ARG0, 
# mul(ARG0, 7.667999712141599))), div(sqrt(mul(sqrt(add(ARG1, add(sqrt(ARG1), 
# add(ARG1, ARG1)))), ARG1)), sub(mul(ARG0, 
# add(sqrt(sqrt(add(sqrt(7.667999712141599), add(ARG1, ARG1)))), 
# sin(sqrt(7.667999712141599)))), 7.667999712141599)))))), div(ARG1, 
# ARG0))))), log10(add(ARG0, sqrt(sqrt(add(ARG0, ARG1))))))