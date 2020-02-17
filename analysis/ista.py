import keras
from keras.constraints import Constraint
import keras.backend as K

class ISTA(Constraint):
    def __init__(self, mu=0.01):
        self.mu = mu

    def __call__(self, w):
        return K.relu( w - self.mu ) - K.relu( -1.0*w - self.mu)

    def get_config(self):
        return {'mu': self.mu}
