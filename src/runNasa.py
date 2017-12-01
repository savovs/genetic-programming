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
=== Weka Exploratory Analysis  ===
Linear Regression Model

effort =

   1645.381  * rely=l,vh +
  -2154.8882 * rely=vh +
    780.4065 * data=l,n,h +
  -1748.3515 * cplx=l,vh,xh +
   2064.9888 * cplx=vh,xh +
    396.2673 * time=vh,h,xh +
   -942.7451 * time=h,xh +
   2279.4264 * time=xh +
  -1181.6677 * pcap=vh,h +
    696.6285 * pcap=h +
    781.9602 * apex=l,vh,h +
   -617.567  * apex=h +
   1010.5671 * ltex=n,h +
   -704.447  * ltex=h +
   -474.0185 * sced=h,n +
      6.2114 * kloc +
     -0.1954 * defects +
     88.5857 * months +
  -2166.5075


=== 10-fold Cross-validation ===

Correlation coefficient                  0.6421
Root mean squared error               1085.1159
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

# Example resulting function from GA:
# mul(sub(ARG1, 6.0425291304844135), sqrt(add(add(div(ARG1, cos(sqrt(mul(ARG1, 
# sqrt(sub(cos(div(ARG0, ARG1)), add(cos(ARG1), sub(sub(ARG0, 
# add(add(6.0425291304844135, ARG0), ARG1)), 1.828653421186695)))))))), 
# add(div(ARG1, cos(sqrt(sub(add(ARG1, 6.0425291304844135), add(ARG0, 
# sub(ARG0, add(sub(ARG0, add(add(ARG1, ARG1), ARG1)), add(ARG1, div(ARG1, 
# cos(sub(ARG0, add(ARG1, ARG1)))))))))))), sub(mul(ARG1, ARG1), 
# sub(cos(add(log10(add(1.68312489879983, ARG0)), 0.9044975227240548)), 
# 1.828653421186695)))), sub(add(ARG0, 
# sub(add(add(add(cos(2.9295970723178715), ARG0), mul(sub(add(ARG1, ARG1), 
# sub(sub(ARG0, add(ARG1, ARG1)), 1.828653421186695)), 6.0425291304844135)), 
# add(add(sub(ARG0, add(ARG1, sub(cos(ARG1), add(cos(ARG1), sub(sub(ARG0, 
# add(add(6.0425291304844135, ARG0), ARG1)), 1.828653421186695))))), add(ARG1, 
# div(ARG1, cos(sub(ARG0, add(ARG1, ARG1)))))), ARG1)), add(ARG1, ARG1))), 
# div(add(mul(ARG1, sqrt(sub(add(sqrt(sub(ARG1, add(div(ARG1, cos(sub(ARG0, 
# add(ARG1, ARG1)))), ARG1))), 6.0425291304844135), add(ARG0, sub(ARG0, 
# add(ARG1, ARG1)))))), sub(sin(add(add(add(add(add(add(add(ARG1, ARG1), 
# sqrt(ARG1)), ARG0), div(mul(2.79977835089251, ARG0), add(4.211402125706582, 
# ARG0))), ARG0), mul(sub(add(ARG1, ARG1), div(add(add(ARG1, ARG1), add(ARG1, 
# ARG1)), ARG0)), ARG0)), add(cos(ARG1), mul(cos(ARG0), div(4.843388281310775, 
# sub(sub(ARG0, ARG0), 1.828653421186695)))))), add(ARG1, add(ARG1, ARG1)))), 
# add(cos(ARG1), 0.9044975227240548))))))