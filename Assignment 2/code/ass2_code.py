#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:05:09 2022

@author: richard
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.optimizers import Adam

rng = np.random.default_rng()


learning_rate = 0.01
epochs = 100
batch_size = 64


def test_train_split(dataframe, training_fraction):
    n_total_examples = len(dataframe)
    n_training_examples = int(n_total_examples*training_fraction)
    
    randomized_dataframe = pd.DataFrame(dataframe).sample(frac=1)
    train_set, test_set = np.split(randomized_dataframe, [n_training_examples])
    return train_set, test_set


def normalize(dataframe):
    data_max = dataframe.max().to_numpy()
    data_min = dataframe.min().to_numpy()
    delta_minmax = data_max - data_min

    normalized_data = (dataframe - data_min) / delta_minmax 
    return normalized_data



# plt.close('all')
# figsize = [12, 8]
# fontsize = 18
# dpi = 200

# pylab.rcParams.update({
#     'figure.figsize': figsize,
#     'legend.fontsize': fontsize,
#     'axes.labelsize': fontsize,
#     'axes.titlesize': fontsize,
#     'xtick.labelsize': fontsize,
#     'ytick.labelsize': fontsize,
#     'savefig.dpi': dpi
# })



zip_datapath = r"../../data/social_capital_zip.csv"

input_labels = ["clustering_zip", "support_ratio_zip"]
predict_labels = ["volunteering_rate_zip", "civic_organizations_zip"]
selected_labels = [*input_labels, *predict_labels]



raw_data = pd.read_csv(zip_datapath)
cleaned_data = raw_data.dropna(subset=selected_labels, inplace=False)

normalized_data = normalize(raw_data)
set_train, set_test = test_train_split(normalized_data, 0.8)

X_train = set_train[input_labels]
X_test = set_test[input_labels]
y_train = set_train[predict_labels]
y_test = set_test[predict_labels]


# building up the model
model = tf.keras.Sequential()
model.add(keras.Input(shape=(2,)))

model.add(layers.Dense(units=64, activation="relu"))
model.add(layers.Dense(units=64, activation="relu"))

model.add(layers.Dense(units=2))

model.compile(
    optimizer=Adam(learning_rate),
    loss='categorical_crossentropy'
)



# model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)






# # plotting population size
# fig = plt.figure(figsize=figsize)
# plt.hist(raw_data["pop2018"], bins=20)
# plt.title('pop2018')
# #plt.savefig(r'../report/images/pop2018_hist.png')



# # plotting histogramms of three chosen datasets
# for id, series in zip(investigated_series_ids, series):
#     fig = plt.figure()
#     plt.hist(series, bins=20)
#     plt.title(id)
#     #plt.savefig(r'../report/images/{}_hist.png'.format(id))



# # plotting a 3D scatter plot to see if 3D data still appears gaussian
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.set_xlabel(investigated_series_ids[0])
# ax.set_ylabel(investigated_series_ids[1])
# ax.set_zlabel(investigated_series_ids[2])

# ax.scatter(ds1, ds2, ds3)
# plt.title("3D scatter plot of the chosen data-series")
# #plt.savefig(r'../report/images/3Dscatter.png')




# # calculating the mean of the chosen 3D data-set
# df = ds1.to_frame().join(ds2).join(ds3)

# mean = df.mean()


