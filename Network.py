import numpy as np 
import Layer
import ActivationFunction
import ErrorFunction

class Network:
    def __init__(self,input_size,output_size,nb_layers,layer_width,possible_answers):
        """constructor"""
        self.__input_size=input_size
        self.__output_size=output_size
        self.__nb_layers=nb_layers
        self.__layer_width=layer_width
        self.__layers=self.generateNetwork()
        self.__possible_answers=possible_answers

    def generateNetwork(self):
        """create a list containing all the layers of the network"""
        matrix = [0 for k in range(2*self.__nb_layers)]
        matrix[0]=Layer.DenseLayer(self.__input_size,self.__layer_width)
        matrix[1]=ActivationFunction.TanH()
        for k in range(1,self.__nb_layers-1):
            matrix[2*k]=Layer.DenseLayer(self.__layer_width,self.__layer_width)
            matrix[2*k+1]=ActivationFunction.TanH()
        matrix[2*self.__nb_layers-2]=Layer.DenseLayer(self.__layer_width,self.__output_size)
        matrix[2*self.__nb_layers-1]=ActivationFunction.TanH()
        return matrix

    def predict(self,input):
        """gives the output of the nn"""
        input=[input]
        output=input
        for layer in self.__layers:
            output=layer.forward(output)
        return output
    
    def decide(self,input):
        """given the answer of the last layer it decides what is the output that is the most likely"""
        prediction=self.predict(input)
        maxi=0
        ecart=ErrorFunction.EQM(prediction,self.__possible_answers[0])
        for k in range(1,len(self.__possible_answers)):
            new_ecart=ErrorFunction.EQM(prediction,self.__possible_answers[k])
            if new_ecart<ecart:
                ecart=new_ecart
                maxi=k
        return self.__possible_answers[maxi]
    
    def backpropag(self,outGrad,learning_rate):
        """back propagate the error gradient"""
        inputGrad=outGrad
        for layer in reversed(self.__layers):
            inputGrad=layer.backward(inputGrad,learning_rate)

    def testSetEQM(self,x_test,y_test):
        errorSum=0
        for k in range(x_test.size):
            prediction=self.predict(x_test[k])
            errorSum+=ErrorFunction.EQM(y_test[k],prediction)
        errorAvg=errorSum/x_test.size
        return errorAvg
        
    def test(self,databaseX,databaseY):
        """when a database with label is given this returns the right answer rate"""
        n=databaseX.size
        errorCount=0
        for k in range(n):
            if self.decide(databaseX[k])!=databaseY[k]:
                errorCount+=1
        errorRate=errorCount/n
        return 1-errorRate
        
    def train(self,epoch,learning_rate,datasetX,datasetY,dataXverif,dataYverif):
        """trains the network with a database"""
        for k in range(epoch):
            errorSum=0
            for j in range(datasetX.size):
                prediction=self.predict(datasetX[k])
                errorSum+=ErrorFunction.EQM(datasetY[k],prediction)
                gradError=ErrorFunction.EQMderiv(datasetY[k],prediction)
                self.backpropag(gradError,learning_rate)
            errorAvg=errorSum/datasetX.size
            print("epoch : "+k+" with average error of "+errorAvg)
        print("training finished and the error on the test database is "+self.testSetEQM(dataXverif,dataYverif))
        
       

        

        