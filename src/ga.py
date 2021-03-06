# Using this dataset: http://openscience.us/repo/effort/cocomo/nasa93.html
# Generate functions that predict effort
import operator, math, random, numpy, os
from deap import base, creator, gp, tools, algorithms
from sklearn.metrics import mean_squared_error


def toolbox_from_pset(pset):
	pset.addPrimitive(operator.sub, 2)
	pset.addPrimitive(operator.mul, 2)
	pset.addPrimitive(operator.add, 2)
	pset.addPrimitive(math.sin, 1)
	pset.addPrimitive(math.cos, 1)

	# Square root any number cheat
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
	def div(x, y):
		try:
			return x / y

		except ZeroDivisionError:
			return 1

	pset.addPrimitive(div, 2)

	# Random range helps speed in fitness improvement
	pset.addEphemeralConstant('floats', lambda: random.uniform(0.01, 10))

	# Aim to minimise fitness
	creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
	creator.create(
		'Individual',
		gp.PrimitiveTree,
		fitness=creator.FitnessMin,
		pset=pset
	)

	toolbox = base.Toolbox()
	toolbox.register(
		'expr',
		gp.genHalfAndHalf,
		pset=pset,
		min_=1,
		max_=2
	)
	
	toolbox.register(
		'individual',
		tools.initIterate,
		creator.Individual,
		toolbox.expr
	)

	toolbox.register('population', tools.initRepeat, list, toolbox.individual)
	toolbox.register('compile', gp.compile, pset = pset)


	toolbox.register('select', tools.selTournament, tournsize = 3)
	toolbox.register('mate', gp.cxOnePoint)
	toolbox.register('expr_mut', gp.genFull, min_ = 0, max_ = 2)
	toolbox.register('mutate', gp.mutUniform, expr = toolbox.expr_mut, pset = pset)

	# Limit the height of the tree to prevent overflow
	toolbox.decorate(
		'mate',
		gp.staticLimit(key = operator.attrgetter('height'), max_value = 17)
	)

	toolbox.decorate(
		'mutate',
		gp.staticLimit(key = operator.attrgetter('height'), max_value = 17)
	)

	return toolbox

# Contain 1 fittest individual
hall_of_fame = tools.HallOfFame(1)

# Perform statistics on fitness
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('min', numpy.min)
