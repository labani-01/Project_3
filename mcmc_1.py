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

 
#print(t)
## ['Apples', 'White Bread', 'Wholemeal Bread']

#print(mV)
## ['A bag of 3 apples', 'A loaf of white bread', 'A loag of wholemeal bread']

 

#plt.plot(t, mV)
#plt.show() 
 
V_0 = float(input("Enter a value for V_0 = "))
A = float(input("Enter a value for A = "))
t_0 = float(input("Enter a value for t_0 = "))
s_0 = float(input("Enter a value for s_0 = "))

 
 
#param = ndarray((4,),float)
#param = np.array([V_0, A, t_0, s_0])
#print(len(param))  

 
def model(t, V_0, A, t_0, s_0):
    u = (t - t_0)/s_0
    model = A*np.exp(-(u + np.exp(u))/2) + V_0
    return model

def loglikelihood(V_0, A, t_0, s_0):
    return - 0.5 * ((mV - model(t, V_0, A, t_0, s_0)) ** 2.0).sum()
 
def lognormpdf(x, mean, sd):
    var = float(sd)**2
    denom = np.log((2*math.pi*var)**.5)
    num = -(float(x)-float(mean))**2/(2*var) 
    return (num + denom)
    
def logprior(V_0, A, t_0, s_0):
    t_0_prior = lognormpdf(t_0, 1000, 1)
    s_0_prior = lognormpdf(s_0, 600, 1) 
    V_0_prior = np.log(uniform.pdf(V_0, loc=-200, scale=1000))
    A_prior = np.log(uniform.pdf(A, loc=-2000, scale=4000))
    return(t_0_prior + s_0_prior + V_0_prior + A_prior)
 
 
def logposterior(V_0, A, t_0, s_0):
    return loglikelihood(V_0, A, t_0, s_0) + logprior(V_0, A, t_0, s_0)
     

    
def logproposal(V_0, A, t_0, s_0):
    p1 = np.log(np.random.normal() + V_0)
    p2 = np.log(np.random.normal() + A)
    p3 = np.log(np.random.normal() + t_0)
    p4 = np.log(np.random.normal() + s_0) 
    return (p1 + p2 + p3 + p4) 
    
def mcmc(a,b,c,d, post, prop, iterations):
    x = [initial]
    p = [post(x[-1])]
    for i in tqdm.tqdm(range(iterations)):
        x_test = prop(x[-1])
        p_test = post(x_test)

        acc = p_test / p[-1]
        u = numpy.random.uniform(0, 1)
        if u <= acc:
            x.append(x_test)
            p.append(p_test)
    return x, p
    
    
chain, prob = mcmc(V_0, A, t_0, s_0, logposterior, logproposal, 1000000)

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
