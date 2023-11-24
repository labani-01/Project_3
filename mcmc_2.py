import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.stats import uniform
import pandas as pd
import csv
import tqdm
from numpy import ndarray

csv_file = 'SingleEventMoyal.csv'

arr_1 = []
arr_2 = []
 
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        arr_1.append(row.get('time'))
        arr_2.append(row.get('data_ch2'))
t = np.array(arr_1, dtype='float64')
mV = np.array(arr_2, dtype='float64') 
plt.scatter(t, mV)
plt.show() 

def model(theta, t):
    u = (t - theta[2])/theta[3]
    return (theta[1] * np.exp(-(u + np.exp(u))/2) + theta[0])
    
def loglikelihood(theta, t, mV):
    return - 0.5 * ((mV - model(theta, t)) ** 2.0).sum()
    
def lognormpdf(x, mean, sd):
    var = float(sd)**2
    denom = np.log((2*math.pi*var)**.5)
    num = -(float(x)-float(mean))**2/(2*var) 
    return (num + denom)
    
#def logprior(theta):
    #t_0_prior = 0
    #s_0_prior = 0
    #A_prior = 0 #flat prior: log(1) = 0
    #V_0_prior = 0
    #return(t_0_prior + s_0_prior + A_prior + V_0_prior)
    
def logprior(theta):
    t_0_prior = lognormpdf(theta[2], 1000, 1)
    s_0_prior = lognormpdf(theta[3], 600, 1) 
    V_0_prior = np.log(uniform.pdf(theta[0], loc=-200, scale=1000))
    A_prior = np.log(uniform.pdf(theta[1], loc=-2000, scale=4000))
    return(t_0_prior + s_0_prior + V_0_prior + A_prior)
 
    
def logposterior(theta, t, mV):
    return logprior(theta) + loglikelihood(theta, t, mV)

def run_mcmc(ln_posterior, nsteps, ndim, theta0, stepsize, args=()):
    """
    Run a Markov Chain Monte Carlo
    
    Parameters
    ----------
    ln_posterior: callable
        our function to compute the posterior
    nsteps: int
        the number of steps in the chain
    theta0: list
        the starting guess for theta
    stepsize: float
        a parameter controlling the size of the , random step
        e.g. it could be the width of the Gaussian distribution
    args: tuple (optional)
        additional arguments passed to ln_posterior
    """
    # Create the array of size (nsteps, ndims) to hold the chain
    # Initialize the first row of this with theta0
    chain = np.zeros((nsteps, ndim))
    chain[0] = theta0
    # Create the array of size nsteps to hold the log-likelihoods for each point
    # Initialize the first entry of this with the log likelihood at theta0
    log_likes = np.zeros(nsteps)
    log_likes[0] = logposterior(chain[0], *args)
    # Loop for nsteps
    for i in tqdm.tqdm(range(1, nsteps)):
        # Randomly draw a new theta from the proposal distribution.
        # for example, you can do a normally-distributed step by utilizing
        # the np.random.randn() function
        #theta_new = chain[i - 1] + stepsize * np.random.randn(ndim)
        theta_new = np.random.normal(ndim) + chain[i - 1]
        # Calculate the probability for the new state
        log_like_new = loglikelihood(theta_new, *args)
        
        # Compare it to the probability of the old state
        # Using the acceptance probability function
        # (remember that you've computed the log probability, not the probability!)
        log_p_accept = log_like_new - log_likes[i - 1]
        
        # Chose a random number r between 0 and 1 to compare with p_accept
        r = np.random.rand()
        
        # If p_accept>1 or p_accept>r, accept the step
        # Else, do not accept the step
        if log_p_accept > np.log(r):
            chain[i] = theta_new
            log_likes[i] = log_like_new
        else:
            chain[i] = chain[i - 1]
            log_likes[i] = log_likes[i - 1]
            
    return chain

chain = run_mcmc(logposterior, 100000, 4, [-1, 2, 10, 5], 1, (t, mV)) 
fig, ax = plt.subplots(4)
ax[0].plot(chain[:, 0])
ax[1].plot(chain[:, 1])
ax[2].plot(chain[:, 2]) 
ax[3].plot(chain[:, 3])
plt.show() 

# Now that we've burned-in, let's get a fresh chain
#chain = run_mcmc(logposterior, 1000000, 4, chain[-1], 0.001, (t, mV))
#fig, ax = plt.subplots(3)
#ax[0].plot(chain[:, 0])
#ax[1].plot(chain[:, 1])
#ax[2].plot(chain[:, 2]) 
#plt.show() 


plt.figure()
plt.title("Evolution of the walker")
plt.plot(chain)
plt.ylabel('t-value')
plt.xlabel('Iteration')

plt.figure()
plt.title("Evolution of the walker")
plt.plot(chain)
plt.xlim(0, 100)
plt.ylabel('t-value')
plt.xlabel('Iteration')

plt.figure()
plt.title("Posterior samples")
_ = plt.hist(chain[1000::1000], bins=100)
plt.show()
