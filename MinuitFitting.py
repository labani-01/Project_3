import csv
import numpy as np
import math
import scipy
from iminuit import Minuit, cost
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import mplhep

def Moyal(x, A, xOff, yOff, sigma):
    
    #The Moyal Function
    
    #Input:
    #x: argument of moyal function
    #A: Constant factor
    #xOff: location of global maximum
    #yOff: Constant y offset
    #sigma: sigma of moyal function

    #Reuturns:
    #y: value of the moyal function at x
    
    u = (x - xOff)/sigma
    y =  A*(np.exp(-(u+np.exp(-u))/2)) + yOff
    return y

def Moyal_Fit(x,y):
    
    #Fits data set to Moyal Function

    #Input:
    #x: x data
    #y: y data

    #Returns:
    #params: parameters of the Moyal function (A, xOff, yOff, sigma)
    
    maxADC = np.max(y)
    maxADC_loc = np.argmax(y)
    c = cost.LeastSquares(x, y, yerror = np.ones(np.asarray(x).shape), model = Moyal)
    m = Minuit(c, A = maxADC*np.exp(-0.5), xOff = maxADC_loc, yOff = 20, sigma = 70)

    #defining bounds of parameters
    m.limits["A"] = (maxADC, maxADC*2)
    m.limits["xOff"] = (maxADC_loc - 100, maxADC_loc + 100)
    m.limits["yOff"] = (-40, 40)
    m.limits["sigma"] = (10, 1000)
    m.migrad()
    m.hesse()
    print(m)
    params = m.values
    
    return params

