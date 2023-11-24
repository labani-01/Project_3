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
x = np.array(arr_1, dtype='float64')
y = np.array(arr_2, dtype='float64') 
u = []

def coffe(theta, x):
    v = []
    s = 1
    for i in range(len(x)):
        v = u.append((x[i] - theta[2])/theta[3]) 
        s = s * (theta[1] * np.exp(-(v + np.exp(v))/2) + theta[0])
    return s
    
 
# assume unit variance normal distributions
def loglikelihood(theta):
    return - 0.5 * ((y - model(theta, x)) ** 2.0).sum()
    
def logprior(theta):
    # Assume unormalized uniform prior
    for p in theta:
        if p > 20000 or p < -20000:
            return - numpy.inf
    return 0
    
def logposterior(theta):
    return coffe(theta, x)
    
    
def proposal(theta):
    return np.random.normal(0, 1, 4) #+ theta
    
#theta = [1, 2, 3, 4]

#z = logposterior(theta)
#print(z) 

 
    
 

def mcmc(theta0, post, prop, iterations):
    chain = np.zeros((iterations, 4))
    chain[0] = theta0
    p = np.zeros(iterations)
    p[0] = post(chain[0])
    for i in tqdm.tqdm(1, iterations):
        theta_test = prop(chain[i - 1])
        p_test = post(theta_test)

        acc = p_test/p[i - 1]
        u = np.random.uniform(0, 1)
        if u <= acc:
            chain[i] = theta_test
            p[i] = p_test
        else:
            chain[i] = chain[i - 1]
            p[i] = p[i - 1]
    return chain, p
    
    
 
chain, prob = mcmc([-1, 0, 10, -5], logposterior, proposal, 1000)
print(chain)

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
_ = plt.hist(chain[100::100], bins=100)
plt.show()
