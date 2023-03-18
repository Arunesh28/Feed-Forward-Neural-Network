import numpy as np

def sigmoid(x):
    return 1. /(1. + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def der_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def der_tanh(x):
    return 1-(np.tanh(x)**2)

def der_relu(x):
    return (x>0)*1

def softmax(x):
    return (np.exp(x)/np.sum(np.exp(x),axis = 0))

def der_softmax(x):
    return softmax(x) * (1-softmax(x))
