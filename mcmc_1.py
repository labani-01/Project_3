import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.stats import uniform


param = [V_0, A, t_0, s_0]
 
 
def model(t, param):
    u = (t - param[2])/param[3] 
    model = param[1]*np.exp(-(u + np.exp(u))/2) + param[0] 
    return model

def loglikelihood(param):
    return - 0.5 * ((V - model(t, param)) ** 2.0).sum()
 
def lognormpdf(x, mean, sd):
    var = float(sd)**2
    denom = np.log((2*math.pi*var)**.5)
    num = -(float(x)-float(mean))**2/(2*var) 
    return (num + denom)
    
def logprior(param):
    t_0_prior = lognormpdf(param[2], 1000, 1)
    s_0_prior = lognormpdf(param[3], 600, 1) 
    V_0_prior = np.log(uniform.pdf(param[0], loc=-200, scale=1000))
    A_prior = np.uniform(uniform.pdf(param[0], loc=-2000, scale=4000))
    return(t_0_prior + s_0_prior + V_0_prior + A_prior)
 
 
def logposterior(param):
    return loglikelihood(param) + logprior(param)
     

     
