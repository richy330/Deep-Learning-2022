# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:50:05 2023

@author: Richard
"""

import os
import pickle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



import numpy as np
import scipy as sp
from numpy.typing import ArrayLike, NDArray
from PIL import Image
import matplotlib.pyplot as plt
plt.close('all')

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


from autoencoders import RotationDecoder


cifar100_image_path_train = r'./data/cifar-100-python/train'
cifar100_image_path_test = r'./data/cifar-100-python/test'


# testplotting image counts
nrows = 3
ncols = 5
angle = 45
create_testplots = True

# encoder hyperparameters
run_rotation_decoder = True
epochs = 10



def unpickle(file_path) -> dict:
    "load cifar-data and return dict containing labels, data and filenames"
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def unflatten_image(arr: ArrayLike) -> NDArray:
    """reshape single RGB image from an array of shape (3072,) into shape (32, 32, 3) 
    as expected by plt.imshow"""
    return arr.reshape((3, 32, 32)).transpose(1, 2, 0)


def unflatten_images(arr: ArrayLike) -> NDArray:
    """reshape multiple RGB images from an array of shape (n, 3072) into shape (n, 32, 32, 3)"""
    return arr.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)


def rotate_image(arr: ArrayLike, angle: float) -> NDArray:
    img_dim = arr.shape[0:2]
    rotated = sp.ndimage.rotate(arr, angle, axes=(1, 0))
    return np.array(Image.fromarray(rotated, mode='RGB').resize(img_dim))


def rotate_images(arr: ArrayLike, angle: float) -> NDArray:
    n_images, *_ = arr.shape
    rotated = np.empty(arr.shape)
    for i in range(n_images):
        rotated[i, ...] = rotate_image(arr[i, ...], angle)
    return rotated


def testplots(nrows, ncols, flatt_imgs):
    #test plots of loaded images
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    plt.suptitle('Original images')
    for i in range(nrows*ncols):
        image = unflatten_image(flatt_imgs[i])
        axs.flatten()[i].imshow(image)
        

    #test plots of rotated images
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    plt.suptitle('Rotated images')
    for i in range(nrows*ncols):
        image = unflatten_image(flatt_imgs[i])
        rotated_image = rotate_image(image, angle)
        
        #sp.ndimage.rotate(image, angle=angle)
        
        axs.flatten()[i].imshow(rotated_image)
    
    
    
    
    
raw_data_train = unpickle(cifar100_image_path_train)['data']
raw_data_test = unpickle(cifar100_image_path_test)['data']



if create_testplots:
    testplots(nrows, ncols, raw_data_train)


#%% rotational decoding
if run_rotation_decoder: 
    x_train = unflatten_images(raw_data_train) / 255
    y_train = rotate_images(x_train, angle) / 255
    
    x_test = unflatten_images(raw_data_test) / 255
    y_test = rotate_images(x_test, angle) / 255
    
    
    img_dim = x_train.shape[1:4]
    rot_auto_enc = RotationDecoder(img_dim, latent_dim=64)
    rot_auto_enc.compile(optimizer='adam', loss=losses.MeanSquaredError())
    rot_auto_enc.fit(
        x_train, x_train,
        epochs=10,
        shuffle=True,
        validation_data=(x_test, y_test))
    
    
    #%% not yet working, probably incorrect architecture
    # plot_img_orig = x_test[0]
    # encoded_imgs = rot_auto_enc.encoder(plot_img_orig)
    # plot_img_decoded = rot_auto_enc.decoder(encoded_imgs)

    # fig, axs = plt.subplots(1, 2)
    # axs.flatten()[0].imshow(plot_img_orig)
    # axs.flatten()[1].imshow(plot_img_decoded)





















