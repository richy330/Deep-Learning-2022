# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 20:14:49 2023

@author: Richard
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model



class RotationDecoder(Model):
    def __init__(self, img_dim, latent_dim):
        super(RotationDecoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),])
        self.decoder = tf.keras.Sequential([
            layers.Dense(np.prod(img_dim), activation='sigmoid'),
            layers.Reshape(img_dim)])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded