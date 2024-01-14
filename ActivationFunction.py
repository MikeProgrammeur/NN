import Layer
import numpy as np

class TanH(Layer.ActivationLayer):
    def __init__(self):
        f_tanh= lambda x : np.tanh(x)
        f_tanhderiv= lambda x: 1-(np.tanh(x))**2
        super().__init__(f_tanh, f_tanhderiv)


class Prelu(Layer.ActivationLayer):
    def __init__(self,alpha):
        f_prelu=lambda x: x if x>0 else alpha*x
        f_preluderiv=lambda x: 1 if x>0 else alpha
        super().__init__(f_prelu, f_preluderiv)