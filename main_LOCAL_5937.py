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
import mcmc_3

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

 
 
#plt.scatter(x, mcmc_3.model(21.94, 1581.99, 1834.32, 73.65))
#plt.scatter(x, y)
#plt.show()

n_iter = 10000
trace, acc = mcmc_3.metropolis(n_iter, (21.70, 1578.50, 1829.32, 72.65), 0.001)
for param, samples in zip(['intercept', 'normalization', 'mean', 'standard_deviation'], trace.T):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2))
    axes[0].plot(samples)
    axes[0].set_ylabel(param)
    axes[1].hist(samples[int(n_iter/2):])
    plt.show()
