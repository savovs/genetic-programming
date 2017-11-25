# Using this dataset: http://openscience.us/repo/effort/cocomo/nasa93.html
# Find columns that predict 'effort'

import operator, math, random, numpy, pandas, os
from deap import base, creator, gp, tools, algorithms

from scipy.io import arff
from sklearn.metrics import mean_squared_error
from pprint import pprint

location = os.path.join(os.getcwd(), os.path.dirname(__file__))
data_path = os.path.join(location, '../nasa93-dem.arff')
# data_path = os.path.realpath(location)


# Parse data
data_blob = arff.loadarff(data_path)
df = pandas.DataFrame(data_blob[0])

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


pset.addEphemeralConstant('ints', lambda: random.randint(1, 100))
pset.renameArguments(ARG0 = 'kloc')
pset.renameArguments(ARG1 = 'months')


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

	# Get mean squared error between predicted and actual effort
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


pop = toolbox.population(n = 200)
hall_of_fame = tools.HallOfFame(1)

multi_stats = tools.MultiStatistics(
	mean_squared_error = tools.Statistics(lambda ind: ind.fitness.values),
	size = tools.Statistics(len)
)

multi_stats.register('avg', numpy.mean)
multi_stats.register('std', numpy.std)
multi_stats.register('min', numpy.min)
multi_stats.register('max', numpy.max)

result, logbook = algorithms.eaSimple(
	pop,
	toolbox,

	cxpb = 0.9,
	mutpb = 0.1,
	ngen = 500,
	stats = multi_stats,
	halloffame = hall_of_fame,
	verbose = True
)

# pprint(hall_of_fame)
