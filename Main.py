import Network
import numpy as np
import mnist

chiffreList= [k for k in range (10)]
possibleOutputs=np.array(chiffreList)
n=Network.Network(784,1,1568,1568,possibleOutputs)

mnist.init()
x_train, y_train, x_test, y_test = mnist.load()
n.train(100,0.1,x_train, y_train, x_test, y_test)
