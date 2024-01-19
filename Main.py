import Network
import numpy as np
import mnist


#test to estimate real function between 0 and 1 
# we ll try to approximate f defined by
def f(x):
    return x**0.5+0.5+np.cos(x)
infinitePossibleAns=[0]
netFun=Network.Network(1,1,32,8,infinitePossibleAns)
x_train=np.random.rand(100)
y_train=np.array([f(x) for x in x_train])
x_test=np.random.rand(50)
y_test=np.array([f(x) for x in x_test])
netFun.train(10,0.1,x_train, y_train, x_test, y_test)



#test MNIST
"""
chiffreList= [k for k in range (10)]
n=Network.Network(784,1,1568,1568,chiffreList)

#mnist.init()
#x_train, y_train, x_test, y_test = mnist.load()
print("mnist loaded")
n.train(10,0.1,x_train, y_train, x_test, y_test)
"""
