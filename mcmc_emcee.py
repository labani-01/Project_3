import numpy as np
import emcee
from MinuitFitting import Moyal

def log_like(x, y, noise, params):
    
    val = Moyal(x, params[0], params[1], params[2], params[3])
    return -0.5*np.sum(((y-val)**2)/(noise**2))

def prior(params):
    prior_A = -0.5 * ((params[0]) / 1.0) ** 2
    prior_xOff = -0.5 * ((params[1]) / 1.0) ** 2
    prior_yOff = -0.5 * ((params[2]) / 1.0) ** 2
    prior_sigma = -0.5 * ((params[3]) / 1.0) ** 2

def log_posterior(x, y, noise, params):
    logPrior = prior(params, paramsGuess)
    if not np.isfinite(logPrior):
        return -np.inf
    return logPrior + log_like(x, y, noise, params)

def mcmc(x, y, noise, nwalkers, paramsGuess):
    dimensions = 4
    A_start = np.random.normal(loc = paramsGuess[0], scale = 5, size = nwalkers)
    xOff_start = np.random.normal(loc = paramsGuess[1], scale = 10, size = nwalkers)
    yOff_start = np.random.normal(loc = paramsGuess[2], scale = 1, size = nwalkers)
    sigma_start = np.random.normal(loc = paramsGuess[3], scale = 2, size = nwalkers)

    start = np.vstack((A_start, xOff_start, yOff_start, sigma_start))

    sampler = emcee.EnsembleSampler(nwalkers, 4, log_posterior, args=(x, y, noise))
    nsteps = 1000
    sampler.run_mcmc(start, nsteps, progress = True)
    samples = sampler.get_chain()

    return samples

fullCSV = np.genfromtxt('SingleEventMoyal.csv', delimiter = ',', skip_header = 1)

x = fullCSV[:,3]
y = fullCSV[:,4]

samples = mcmc(x, y, 10, 10, [1580, 1834, 22, 74])

plt.figure(figsize=(10, 6))
plt.plot(samples[:, :, 0], color='k', alpha=0.3)
plt.ylabel('a')
