import csv
import numpy as np
import math
import scipy
from iminuit import Minuit, cost
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import mplhep

from MinuitFitting import Moyal, Moyal_Fit

#First need to read the file
fullCSV = np.genfromtxt('SingleEventMoyal.csv', delimiter = ',', skip_header = 1)

x = fullCSV[:,3]
y = fullCSV[:,4]

#Now we can get expected fit parameters using Minuit's optimization method

params = Moyal_Fit(x, y)

print(params)
y_fitted = Moyal(x, *params)

mplhep.style.use("LHCb2")
fig, axes = plt.subplots()
axes.plot(x,y, color = 'red')
axes.plot(x, y_fitted, color = 'blue')
fig.set_size_inches(10, 7)
plt.show()


