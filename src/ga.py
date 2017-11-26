# Using this dataset: http://openscience.us/repo/effort/cocomo/nasa93.html
# Find columns that predict 'effort'

import pandas as pd
import operator, math, random, numpy, os
from deap import base, creator, gp, tools, algorithms

from scipy.io import arff
from sklearn.metrics import mean_squared_error
from pprint import pprint

location = os.path.join(os.getcwd(), os.path.dirname(__file__))
data_path = os.path.join(location, '../nasa93-dem.arff')
# data_path = os.path.realpath(location)


# Parse data
data_blob = arff.loadarff(data_path)
df = pd.DataFrame(data_blob[0])

# Nominal columns contain arbitrary information about categories
# so they won't be useful to determine 'effort'

# Running Weka linear regression on the dataset shows that
# the 'kloc' and 'months' columns can be used to determine 'effort'
data = df[['kloc', 'months', 'effort']].copy()

kloc = data[['kloc']].values.flatten().tolist()
months = data[['months']].values.flatten().tolist()
effort = data[['effort']].values.flatten().tolist()


# Create primitive set
pset = gp.PrimitiveSet('EFFORT', 2)

pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)

# Square root any number
def sqrt(x):
	if x >= 0:
		return operator.pow(x, 0.5)

	if x < 0:
		return operator.pow(abs(x), 0.5)

pset.addPrimitive(sqrt, 1)

# Prevent math domain error
def log10(x):
	if x <= 0.0:
		return 0

	return math.log10(x)

pset.addPrimitive(log10, 1)

# Prevent division by zero
def division(x, y):
	try:
		return x / y

	except ZeroDivisionError:
		return 1

pset.addPrimitive(division, 2)


pset.addEphemeralConstant('floats', lambda: random.uniform(0.01, 10))
pset.renameArguments(ARG0 = 'kloc')
pset.renameArguments(ARG1 = 'months')

# Aim to minimise fitness
creator.create('FitnessMin', base.Fitness, weights = (-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness = creator.FitnessMin, pset = pset)

toolbox = base.Toolbox()
toolbox.register('expr', gp.genHalfAndHalf, pset = pset, min_ = 1, max_ = 2)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gp.compile, pset = pset)

def fitness(individual):
	# Transform the tree expression in a callable function
	func = toolbox.compile(expr = individual)

	funcResults = []

	for loc, month in zip(kloc, months):
		funcResults.append(func(loc, month))

	# Get mean squared error between actual and predicted effort
	mse = mean_squared_error(
		effort,
		funcResults
	)

	twoPointPrecision = '%.2f' % mse
	return float(twoPointPrecision),


toolbox.register('evaluate', fitness)
toolbox.register('select', tools.selTournament, tournsize = 3)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('expr_mut', gp.genFull, min_ = 0, max_ = 2)
toolbox.register('mutate', gp.mutUniform, expr = toolbox.expr_mut, pset = pset)

# Limit the height of the tree
toolbox.decorate('mate', gp.staticLimit(key = operator.attrgetter('height'), max_value = 17))
toolbox.decorate('mutate', gp.staticLimit(key = operator.attrgetter('height'), max_value = 17))


pop = toolbox.population(n = 100)

# Hall of fame contains only 1 individual
hall_of_fame = tools.HallOfFame(1)

# Perform statistics on fitness
stats = tools.Statistics(lambda ind: ind.fitness.values)

stats.register('avg', numpy.mean)
stats.register('std', numpy.std)
stats.register('min', numpy.min)
stats.register('max', numpy.max)

results = []

for i in range(3):
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

	results.append({
		'best': {
			'fitness': toolbox.evaluate(hall_of_fame[0]),
			'primitive': hall_of_fame[0]
		},

		'stats': logbook.select('gen', 'avg', 'min', 'max')
	})
