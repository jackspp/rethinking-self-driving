"""
Utils
"""

from keras.engine.topology import Layer
import keras.backend as K


class PsudoSequential(object):
    def __init__(self,):
        self.layer_list = []

    def add(self, layer):
        self.layer_list.append(layer)

    def __call__(self, keras_tensor):
        k_tensor = keras_tensor
        for layer in self.layer_list:
            k_tensor = layer(k_tensor)

        return k_tensor



class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def compute_output_shape(self, input_shape):
	return input_shape
