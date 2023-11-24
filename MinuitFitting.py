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
    u = (x - xOff)/sigma
    y =  A*(np.exp(-(u+np.exp(-u))/2)) + yOff
    return y

def Moyal_Fit(x,y):
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
    
    return m.values

