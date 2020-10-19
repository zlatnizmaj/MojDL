"""Implementation of vanila autoencoder in TensorFlow 2.0 Subclassing API"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Nam Ng"

from functools import  partial
import  tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.sigmoid
        )

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)
