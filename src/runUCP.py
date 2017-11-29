# Dataset: http://openscience.us/repo/effort/effort-other/ucp.html

import os
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold
from deap import algorithms, gp
from operator import itemgetter
from numpy import mean, std
from matplotlib import pyplot as plt

from ga import toolbox_from_pset, stats, hall_of_fame
from fitness import fitness

# Explore data, find fitness goals to meet:

# === Weka Exploration of Data ===
# Attributes:   15
#               Simple Actors
#               Average Actors
#               Complex Actors
#               UAW
#               Simple UC
#               Average UC
#               Complex UC
#               UUCW
#               TCF
#               ECF
#               Real_Effort_Person_Hours
#               Sector
#               Language
#               Methodology
#               ApplicationType
# Test mode:    10-fold cross-validation

# === Linear Regression Model ===

# Real_Effort_Person_Hours =

#    -340.4336 * Complex Actors +
#      45.4729 * Complex UC +
#     304.2088 * Sector=Service Industry,Banking,Professional Services,Wholesale & Retail,Electronics & Computers,Communication,Manufacturing +
#    -880.3201 * Sector=Communication,Manufacturing +
#    1152.557  * Sector=Manufacturing +
#    -862.2697 * Language=DELPHI,Visual Basic,SQL,C#,Oracle,C,Java,XML,C++,.Net,CSP +
#     424.522  * Language=C#,Oracle,C,Java,XML,C++,.Net,CSP +
#    1161.8863 * Language=Oracle,C,Java,XML,C++,.Net,CSP +
#   -1065.6929 * Language=C,Java,XML,C++,.Net,CSP +
#    1223.438  * Language=CSP +
#   -1187.4421 * Methodology=Multifunctional Teams,Incremental,Waterfall,Unified Process and its variants,Rapid Application Development,Personal Software Process (PSP) +
#    1225.1163 * Methodology=Incremental,Waterfall,Unified Process and its variants,Rapid Application Development,Personal Software Process (PSP) +
#    1028.057  * Methodology=Rapid Application Development,Personal Software Process (PSP) +
#     706.6409 * ApplicationType=Mathematically-Intensive Application,Real-Time Application,Real-Time application +
#   -1137.9839 * ApplicationType=Real-Time Application,Real-Time application +
#    2117.1876 * ApplicationType=Real-Time application +
#    7045.5009

# Correlation coefficient                  0.1599
# Root mean squared error                860.7491 <------- Linear Regression Baseline


# === Gaussian Processes ===

# Kernel used:
#   Linear Kernel: K(x,y) = <x,y>

# All values shown based on: Normalize training data

# Average Target Value : 0.3627642850267894
# Inverted Covariance Matrix:
#     Lowest Value = -0.3366840186206354
#     Highest Value = 0.8904301710499111
# Inverted Covariance Matrix * Target-value Vector:
#     Lowest Value = -0.5434966830795906
#     Highest Value = 0.5262426079375616
 

# Correlation coefficient                  0.3372
# Root mean squared error                658.8125 <------- Gaussian Processes Baseline




location = os.path.join(os.getcwd(), os.path.dirname(__file__))
data_path = os.path.join(location, '../UCP_Dataset.csv')

df = pd.read_csv(data_path, sep=';')

# Drop irrelevant columns
df.drop(['Project_No', 'DataDonator', 'Real_P20'], 1, inplace=True)

# Fill empty values with zeros
df = df.fillna(0)


# The GA needs numbers, so we need to
# replace categorical columns with dummie columns with binary numbers for each category
# i.e. column 'Sector' will be replaced with 'Sector_value1', 'Sector_value2'...

df = pd.get_dummies(df, columns=['Sector', 'Language', 'Methodology', 'ApplicationType'])

# Create primitive set with arity equal to number of columns without effort 
pset = gp.PrimitiveSet('EFFORT', df.shape[1] - 1)
toolbox = toolbox_from_pset(pset)

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
		tb=toolbox,
		dataRows=trainingDfWithoutEffort.values.tolist(),
		efforts=trainingDf[['Real_Effort_Person_Hours']].values.flatten().tolist()
	)

	pop = toolbox.population(n=200)

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

	plt.plot(logbook.select('min'), color='gray', alpha=0.15)
	# training_best_of_gen_errors.append(logbook.select('min'))

	# Test GA, register fitness function with testing data
	toolbox.register(
		'evaluate',
		fitness,
		tb=toolbox,
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
print('GA mean rmse: %0.2f (+/- %0.2f)' % (errors_mean, errors_std))


# Weka root mean squared error baselines
plt.axhline(860.7491, label='Linear Regression Baseline', color='#E46161')
plt.axhline(658.8125, label='Gaussian Processes Baseline', color='#F1B963')

# Result
plt.axhline(
	errors_mean,
	label='Genetic Algorithm: %0.2f (+/- %0.2f)' % (errors_mean, errors_std),
	color='#A3DE83'
)

plt.xlabel('Generation')
plt.ylabel('Root Mean Square Error')
plt.title('UCP dataset')
plt.suptitle('Effort Estimation Cross-validation')
plt.legend()
plt.show()