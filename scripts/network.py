#! /usr/bin/env python
#################################################################################
#     File Name           :     network.py
#     Created By          :     yang
#     Creation Date       :     [2017-01-26 21:54]
#     Last Modified       :     [2017-01-26 22:37]
#     Description         :      
#################################################################################
import lasagne, theano
import theano.tensor as T
from layer import FeatureProjectionLayer, DiagMaskLayer, MeanPooling_1D_Length_Layer, NormalizedLayer
import cPickle
import numpy as np

class Network(object):
    def __init__(self):
        self.network = None

    def save_to_file(self, file_name):
        parameter_list = lasagne.layers.get_all_param_values(self.network)
        with open(file_name, "w") as fout:
            cPickle.dump(parameter_list, fout)

    def load_from_file(self, file_name):
        with open(file_name) as fin:
            parameter_list = cPickle.load(fin)
        lasagne.layers.set_all_param_values(self.network, parameter_list)

class NetworkLen(Network):
    def __init__(self):
        super(NetworkLen, self).__init__()
        self.embedding_func = None

    def build_theano_embedding_function(self, deterministic=None):
        input_tensor = T.tensor3("input_tensor", dtype="float32")
        length_tensor = T.ivector("length_tensor")
        embedding = lasagne.layers.get_output(self.network, {
            self.input_layer: input_tensor,
            self.length_layer: length_tensor
        }, deterministic=deterministic)
        func = theano.function([input_tensor, length_tensor], embedding, updates=None)
        return func

    def get_embedding(self, distance_matrix):
        embedding_size = self.network.output_shape[-1]
        length = distance_matrix.shape[0]
        distance_matrix[range(length), range(length)] = float("inf")
        self.distance_matrix_buf[:, :] = float("inf")
        self.distance_matrix_buf[:length, :length] = distance_matrix[:, :]
        if self.embedding_func is None:
            self.embedding_func = self.build_theano_embedding_function(deterministic=True)
        return self.embedding_func(self.distance_matrix_buf[None, :, :], np.array([length], dtype = 'int32'))[0]

class DeepFold(NetworkLen):
    def __init__(self, max_length=256, projection_level=3):
        self.distance_matrix_buf = np.zeros((max_length, max_length), dtype = 'float32')
        super(DeepFold, self).__init__()
        self.input_layer = lasagne.layers.InputLayer(shape=(None, max_length, max_length))
        self.length_layer = lasagne.layers.InputLayer(shape=(None,))
        feature_layer = FeatureProjectionLayer(self.input_layer, projection_level=projection_level)
        filter_number_list = [128, 256, 512, 512, 512, 398]
        filter_size_list = [12, 4, 4, 4, 4, 4]
        for filter_number, filter_size in zip(filter_number_list, filter_size_list):
            feature_layer = lasagne.layers.Conv2DLayer(feature_layer,
                                                       num_filters=filter_number,
                                                       filter_size=(filter_size, filter_size),
                                                       pad=int(filter_size / 2 - 1),
                                                       stride=2,
                                                       nonlinearity=lasagne.nonlinearities.linear)
            feature_layer = lasagne.layers.BatchNormLayer(feature_layer)
            feature_layer = lasagne.layers.NonlinearityLayer(feature_layer, nonlinearity=lasagne.nonlinearities.rectify)
            feature_layer = lasagne.layers.DropoutLayer(feature_layer, p=0.5)
        
        feature_layer = DiagMaskLayer(feature_layer)
        feature_layer = MeanPooling_1D_Length_Layer(incomings=[feature_layer, self.length_layer],
                                                    factor=2 ** (len(filter_number_list)))
        feature_layer = NormalizedLayer(feature_layer)
        self.network = feature_layer


