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
from mcmc_emcee import log_like, prior, log_posterior, mcmc
import mcmc_handwritten
import emcee
import auto_corr
 

#First need to read the file
fullCSV = np.genfromtxt('SingleEventMoyal.csv', delimiter = ',', skip_header = 1)

x = fullCSV[:,3]
y = fullCSV[:,4]
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('time(sec)')
plt.ylabel('Voltage(mV)')
plt.show()
 

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
plt.xlabel('time(sec)')
plt.ylabel('Voltage(mV)')
fig.set_size_inches(10, 7)
plt.show()



#############################################################################
#Distribution of parameters from the handwritten mcmc
#############################################################################
 
n_iter = 10000
trace, acc = mcmc_handwritten.metropolis(n_iter, (20, 1580, 1830, 72.98), 0.01)
for param, samples in zip(['offset voltage (V_0)', 'amplitude (A)', 'mean', 'width'], trace.T):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2))
    axes[0].plot(samples)
    axes[0].set_xlabel('iterations')
    axes[0].set_ylabel(param)
    axes[1].hist(samples[int(n_iter/2):])
    plt.show()
 
 
 
#############################################################################
#Now look at off the shelf method (iteration = 1000)
#############################################################################

samples = mcmc(x,y,10,10, [1580, 1830, 22, 74])

plt.figure(figsize=(10, 6))
plt.plot(samples[:, :, 0], color='k', alpha=0.3)
plt.xlabel('iterations')
plt.ylabel('A')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(samples[:, :, 1], color='k', alpha=0.3)
plt.xlabel('iterations')
plt.ylabel('xOff')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(samples[:, :, 2], color='k', alpha=0.3)
plt.xlabel('iterations')
plt.ylabel('yOff')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(samples[:, :, 3], color='k', alpha=0.3)
plt.xlabel('iterations')
plt.ylabel('width')
plt.show()

#######################################################################
#Auto-correlation time using the same off shelf method with 10000 iterations
# and 50 walkers
#######################################################################

samples_1, tau = auto_corr.mcmc(x,y,10,10, [1580, 1830, 22, 74])
print("Auto-correlation time", tau)
