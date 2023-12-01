import numpy as np
import emcee
from MinuitFitting import Moyal
import matplotlib.pyplot as plt

def log_like(x, y, noise, params):
    """The log_like Function
    
    Input:
    x: x values
    y: y values
    noise: standard deviation of the noise present in data set
    params: parameters of the Moyal function (A, xOff, yOff, sigma)
     
    Reuturns:
    
    
    
    """
    
    val = Moyal(x, params[0], params[1], params[2], params[3])
    return -0.5*np.sum(((y-val)**2)/(noise**2))

def prior(params):
    """The prior Function
    
    Input:
    params: parameters of the Moyal function (A, xOff, yOff, sigma)
     
    Reuturns:
    prior: calculating sum of log-priors of each parameters 
    
    
    """
    
    A, xOff, yOff, sigma = params
    prior_A = -0.5 * ((A)-1581 / 1) ** 2
    prior_xOff = -0.5 * ((xOff)-1834 / 1.0) ** 2
    prior_yOff = -0.5 * ((yOff)-22 / 1.0) ** 2
    prior_sigma = -0.5 * ((sigma)-73 / 1.0) ** 2
    return prior_A + prior_xOff + prior_yOff + prior_sigma
    
def log_posterior(params, x, y, noise):
    """The log_posterior Function
    
    Input:
    x: x values
    y: y values
    noise: standard deviation of the noise present in data set
    params: parameters of the Moyal function (A, xOff, yOff, sigma)
     
    Reuturns:
    log-posterior: logPrior + log_likelihood
    
    
    """
    logPrior = prior(params)
    if not np.isfinite(logPrior):
        return -np.inf
    return logPrior + log_like(x, y, noise, params)

def mcmc(x, y, noise, nwalkers, paramsGuess):
    """The mcmc Function
    
    Input:
    x: x values
    y: y values
    noise: standard deviation of the noise present in data set
    nwalkers: number of initial conditions we want to set
    paramsGuess: initial parameters of the Moyal function (A, xOff, yOff, sigma)
     
    Reuturns:
    samples: chain of values for the parameters in the parameter space
    tau: autocorrelation-time
    
    """
    
    dimensions = 4
    A_start = np.random.normal(loc = paramsGuess[0], scale = 5, size = nwalkers)
    xOff_start = np.random.normal(loc = paramsGuess[1], scale = 10, size = nwalkers)
    yOff_start = np.random.normal(loc = paramsGuess[2], scale = 1, size = nwalkers)
    sigma_start = np.random.normal(loc = paramsGuess[3], scale = 2, size = nwalkers)
    start = []
    for i in range(nwalkers):
        start.append([A_start[i], xOff_start[i], yOff_start[i], sigma_start[i]])
    start = np.asarray(start)
        
    

    sampler = emcee.EnsembleSampler(nwalkers, 4, log_posterior, args=(x, y, noise))
    nsteps = 10000
    sampler.run_mcmc(start, nsteps, progress = True)
    samples = sampler.get_chain()
    tau = sampler.get_autocorr_time()
    return samples, tau
    

