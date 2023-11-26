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

#############################################################################
#First we can get expected fit parameters using Minuit's optimization method
#############################################################################

params_opt = Moyal_Fit(x, y) #params generated using Minuit Optimization
y_opt = Moyal(x, *params_opt) #y data generated using above params

print('Parameters generated using Minuit Optimization:', params_opt)

mplhep.style.use("LHCb2") 
fig, axes = plt.subplots()
axes.plot(x,y, color = 'red')
axes.plot(x, y_opt, color = 'blue')
fig.set_size_inches(10, 7)
plt.show()

#############################################################################
