#! /usr/bin/env python
#################################################################################
#     File Name           :     layer.py
#     Created By          :     yang
#     Creation Date       :     [2017-01-26 22:15]
#     Last Modified       :     [2017-01-26 22:22]
#     Description         :     Customized Layers
#################################################################################
import lasagne
import theano.tensor as T
import numpy as np
import theano

class FeatureProjectionLayer(lasagne.layers.Layer):
    def __init__(self, incoming, projection_level, **kwargs):
        super(FeatureProjectionLayer, self).__init__(incoming, **kwargs)
        self.projection_level = projection_level
        # self.W = self.add_param(lasagne.init.Constant(0.1), (self.projection_level,), name="W")

    def get_output_for(self, input, **kwargs):
        res = [(input ** (- i * 2 - 2)).dimshuffle(0, 'x', 1, 2) for i in range(self.projection_level)]
        res = T.concatenate(res, axis=1)
        # return T.tensordot(res, self.W, [[1], [0]]).dimshuffle(0, 'x', 1, 2)
        return res

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.projection_level, input_shape[1], input_shape[2]

class DiagMaskLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        def make_diag(n):
            return np.identity(n, dtype='float32')

        super(DiagMaskLayer, self).__init__(incoming, **kwargs)
        self.mask = make_diag(incoming.output_shape[-1])

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]

    def get_output_for(self, input, **kwargs):
        return T.sum(input * self.mask[None, None, :, :], axis=-1)

class MeanPooling_1D_Length_Layer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, factor, **kwargs):
        super(MeanPooling_1D_Length_Layer, self).__init__(incomings, **kwargs)
        self.factor = factor

    def get_output_shape_for(self, input_shapes, **kwargs):
        return input_shapes[0][0], input_shapes[0][1]

    def get_output_for(self, inputs, **kwargs):
        distance = inputs[0]
        length = (inputs[1] + self.factor - 1) // self.factor

        def mean_value(distance_piece, l):
            return T.mean(distance_piece[:, :l], axis=-1)

        mean_result, _ = theano.scan(fn=mean_value,
                                     outputs_info=None,
                                     sequences=[distance, length],
                                     non_sequences=None)
        return mean_result


class NormalizedLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(NormalizedLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        norm = T.sqrt(T.sum(input[:, :] * input[:, :], axis=1))[:, None]
        return input[:, :] / norm[:, :]

