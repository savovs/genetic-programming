from sklearn.metrics import mean_squared_error
from math import sqrt

def fitness(individual, tb = None, dataRows = [], efforts = []):
	# Compile the tree into a callable function
	compiledFunction = tb.compile(expr=individual)

	funcResults = []

	for row in dataRows:
		# Unpack each row and use column values as args
		funcResults.append(compiledFunction(*row))

	# Get root mean squared error between actual and predicted effort
	rmse = sqrt(
		mean_squared_error(
			efforts,
			funcResults
		)
	)

	twoPointPrecision = '%.2f' % rmse
	return float(twoPointPrecision),