import numpy as np

def EQM(ytrue,ypred):
    return np.mean(np.power(ytrue-ypred,2))

def EQMderiv(ytrue,ypred):
    return 2 * (ypred-ytrue)/np.size(ytrue)