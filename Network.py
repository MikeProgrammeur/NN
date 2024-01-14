import numpy as np 
import Layer
import ActivationFunction

class Network():
    def __init__(self,input_size,output_size,nb_layers,layer_width):
        self.__input_size=input_size
        self.__output_size=output_size
        #self.__activ_fun=activ_fun
        self.__nb_layers=nb_layers
        self.__layer_width=layer_width
        self.__layers=self.generateNetwork(self)

    def generateNetwork(self):
        matrix = np.zeros(self.__nb_layers*2)
        matrix[0]=Layer.DenseLayer(self.__input_size,self.__layer_width)
        matrix[1]=ActivationFunction.TanH()
        for k in range(1,self.__nb_layers-1):
            matrix[2*k]=Layer.DenseLayer(self.__layer_width,self.__layer_width)
            matrix[2*k+1]=ActivationFunction.TanH()
        matrix[2*self.__nb_layers-2]=Layer.DenseLayer(self.__layer_width,self.__output_size)
        matrix[2*self.__nb_layers-1]=ActivationFunction.TanH()
            

    def predict(self,input):
        pass


    def train(self,epoch):
        pass

        