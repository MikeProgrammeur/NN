import math
import numpy as np 

class Layer():
    def __init__(self):
        self.__input=None
        self.__output=None
    
    def forward(self,input):
        "gives the output of the layer for a certain input"
        pass
    
    def backward(self,output_gradient,learning_rate):
        "gives the input error gradient for a certain output gradient and update parameters "
        pass




class DenseLayer(Layer):
    def __init__(self,input_size,output_size):
        self.__weight=np.random.randn(output_size,input_size)
        self.__bias=np.random.randn(output_size,1)
    
    def forward(self,input):
        "gives the output of the layer for a certain input"
        self.__input=input
        return self.__weight@input+self.__bias
    
    def backward(self,output_gradient,learning_rate):
        "gives the input error gradient for a certain output gradient and update W and the bias "
        #updating bias and weight
        self.__bias-= learning_rate*output_gradient
        self.__weight-=learning_rate*output_gradient@np.transpose(input)
        return np.transpose(self.__weight)@output_gradient


class ActivationLayer(Layer):
    def __init__(self,activation,activationderiv):
        self.__activation=activation
        self.__activationderiv=activationderiv

    def forward(self, input):
        self.__input=input
        return self.__activation(input)
    
    def backward(self,output_gradient,learning_rate):
        return np.multiply(output_gradient,self.__activationderiv(input))