import numpy as np
import pymc3 as pm
from MinuitFitting import Moyal, Moyal_Fit
import matplotlib.pyplot as plt

def pymc3_Fit(x, y, A, xOff, yOff, sigma):

    with pm.Model() as model:
        
        A_distr = pm.Normal('A', mu = A, sd = 1)
        xOff_distr = pm.Normal('xOff', mu = xOff, sd = 1)
        yOff_distr = pm.Normal('yOff', mu = yOff, sd = 1)
        sigma_distr = pm.Normal('sigma', mu = sigma, sd = 1)

        y_guess = Moyal(x, A_distr, xOff_distr, yOff_distr, sigma_distr)
        y_new = pm.Normal('y_guess', mu=y_guess, sd = 10, observed = y)
    
        trace = pm.sample(2000, tune=10000)
    
    return trace
    
