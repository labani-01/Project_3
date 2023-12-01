import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.stats import uniform
import pandas as pd
import csv
import tqdm
from numpy import ndarray
from scipy.stats import gamma, norm
 
 
csv_file = 'SingleEventMoyal.csv'

arr_1 = []
arr_2 = []
 
 
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        arr_1.append(row.get('time'))
        arr_2.append(row.get('data_ch2'))
x = np.array(arr_1, dtype='float64')
y = np.array(arr_2, dtype='float64') 
 
 
 
 
dgamma = gamma.logpdf
dnorm = norm.logpdf

def calc_posterior(a, b, t, c):
    """The calc_posterior Function
    
    Input:
    a: constant y offset
    b: Constant factor
    t: location of global maximum
    c: width of Moyal function
     
    Reuturns:
    logp: calculating log of posterior distribution
    
    
    """
    
    
    logp = dnorm(a, 0, 10000) + dnorm(b, 0, 10000) + dnorm(t, 0, 10000) + dnorm(c, 0, 10000) #sum of log of priors 
    u = (x - t)/c
    mod = b * np.exp(-(u + np.exp(-u))/2) + a #Moyal function
    # Data likelihood
    logp += sum(dnorm(y - mod, 0, 10)) 
    return logp
    
    
rnorm = np.random.normal
runif = np.random.rand
np.random.seed(42)

def metropolis(n_iterations, initial_values, prop_var=1):
    """ The metropolis Function
    
    Input:
    n_iterations: total number of iterations
    initial_values: starting values of each parameters
    
    Reuturns:
    trace: Chain of values for the parameters, row = n_iteraion, column = no. of parameters
    accepted: Acceptance counts
    
    
    """
    
    n_params = len(initial_values)
            
    # Initial proposal standard deviations
    prop_sd = [prop_var]*n_params  #create a list of four elements, each of which is the integer 1
    
    # Initialize trace for parameters
    trace = np.empty((n_iterations+1, n_params))
    
    # Set initial values
    trace[0] = initial_values
        
    # Calculate joint posterior for initial values
    current_log_prob = calc_posterior(*trace[0])
    
    # Initialize acceptance counts
    accepted = [0]*n_params
    
    for i in tqdm.tqdm(range(n_iterations)):
        # Grab current parameter values
        current_params = trace[i]
    
        for j in range(n_params):
    
            # Get current value for parameter j
            p = trace[i].copy()
    
            # Propose new value
            if j==2 or j==3:
                # Ensure mean and standard-deviation are positive
                theta = np.exp(rnorm(np.log(current_params[j]), prop_sd[j]))
            else:
                theta = rnorm(current_params[j], prop_sd[j])
            
            # Insert new value 
            p[j] = theta
    
            # Calculate log posterior with proposed value
            proposed_log_prob = calc_posterior(*p)# Log-acceptance rate
            alpha = proposed_log_prob - current_log_prob
    
            # Sample a uniform random variate
            u = runif()
    
            # Test proposed value
            if np.log(u) < alpha:
                # Accept
                trace[i+1,j] = theta
                current_log_prob = proposed_log_prob
                accepted[j] += 1
            else:
                # Reject
                trace[i+1,j] = trace[i,j]
                
    return trace, accepted


 
 
 
