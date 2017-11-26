from matplotlib import pyplot as plt
from pprint import pprint
import pandas as pd

from ga import results


print(results)
# Put statistics in lists
statVectors = {}
for key in results[0]['stats']:
	statVectors[key] = []


for item in results:
	for key in item['stats']:
		statVectors[key].append(item['stats'][key])

pprint(statVectors)