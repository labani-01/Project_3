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
ev = []
 
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        arr_1.append(row.get('time'))
        arr_2.append(row.get('data_ch2'))
x = np.array(arr_1, dtype='float64')
y = np.array(arr_2, dtype='float64') 
 
def model(a, b, t, c):
    u = (x - t)/c
    mod = b * np.exp(-(u + np.exp(-u))/2) + a
    return mod
 
dgamma = gamma.logpdf
dnorm = norm.logpdf

def calc_posterior(a, b, t, c):
    # Calculate joint posterior, given values for V_0, A, t, sigma_0

    # Priors on a,b
    #t_0_prior = dnorm(t, 1000, 1)
    #s_0_prior = dnorm(c, 600, 1) 
    #V_0_prior = np.log(uniform.pdf(a, loc=-200, scale=1000))
    #A_prior = np.log(uniform.pdf(b, loc=-2000, scale=4000))
    #logp = t_0_prior + s_0_prior + V_0_prior + A_prior
    logp = dnorm(a, 0, 10000) + dnorm(b, 0, 10000) + dnorm(t, 0, 10000) + dnorm(c, 0, 10000)
    #logp = dnorm(a, 21.94, 500) + dnorm(b, 1581.98, 500) + dnorm(t, 1834.32, 500) + dnorm(c, 73.65, 500)
    # Calculate mu
    u = (x - t)/c
    mod = b * np.exp(-(u + np.exp(-u))/2) + a
    # Data likelihood
    logp += sum(dnorm(y - mod, 0, 1)) 
    return logp
    
    
rnorm = np.random.normal
runif = np.random.rand
np.random.seed(42)

def metropolis(n_iterations, initial_values, prop_var=1):

    n_params = len(initial_values)
            
    # Initial proposal standard deviations
    prop_sd = [prop_var]*n_params
    
    # Initialize trace for parameters
    trace = np.empty((n_iterations+1, n_params))
    
    # Set initial values
    trace[0] = initial_values
        
    # Calculate joint posterior for initial values
    current_log_prob = calc_posterior(*trace[0])
    
    # Initialize acceptance counts
    accepted = [0]*n_params
    
    for i in range(n_iterations):
    
        if not i%1000: print('Iteration %i' % i)
    
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


 
    
 
