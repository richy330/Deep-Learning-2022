#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:05:09 2022

@author: richard
"""

# required hack because of collision of .dlls on windows
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from keras.losses import log_cosh, mean_squared_error

rng = np.random.default_rng()


learning_rate = 0.0001
epochs = 50
batch_size = 64
cost_function = mean_squared_error
optimizer = Adam(learning_rate)






#%% helper functions

def train_test_split(dataframe, training_fraction):
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




#%% loading and preparing datasets
zip_datapath = r"../../data/social_capital_zip.csv"

input_labels = [
    'ec_zip', 'ec_se_zip',
    'nbhd_ec_zip', 'ec_grp_mem_zip', 'ec_high_zip', 'ec_high_se_zip',
    'nbhd_ec_high_zip', 'ec_grp_mem_high_zip', 'exposure_grp_mem_zip',
    'exposure_grp_mem_high_zip', 'nbhd_exposure_zip', 'bias_grp_mem_zip',
    'bias_grp_mem_high_zip', 'nbhd_bias_zip', 'nbhd_bias_high_zip',
    'clustering_zip', 'support_ratio_zip'
]

predict_labels = ["volunteering_rate_zip", "civic_organizations_zip"]


selected_labels = [*input_labels, *predict_labels]



raw_data = pd.read_csv(zip_datapath)
selected_data = raw_data[selected_labels]
cleaned_data = selected_data.dropna()

normalized_data = normalize(cleaned_data)
set_train, set_test = train_test_split(normalized_data, 0.8)
set_train, set_validation = train_test_split(set_train, 0.8)



X_train = set_train[input_labels]
X_validate = set_validation[input_labels]
X_test = set_test[input_labels]

y_train = set_train[predict_labels]
y_validate = set_validation[predict_labels]
y_test = set_test[predict_labels]


#%% building and training the models


model1 = tf.keras.Sequential()
model2 = tf.keras.Sequential()
model3 = tf.keras.Sequential()
model4 = tf.keras.Sequential()


model1.add(keras.Input(shape=(len(input_labels),)))

model1.add(layers.Dense(units=8, activation="relu"))
model1.add(layers.Dense(units=8, activation="relu"))
# model1.add(layers.Dense(units=16, activation="sigmoid"))

model1.add(layers.Dense(units=len(predict_labels)))

model1.compile(
    optimizer=optimizer,
    loss=cost_function
)




model2.add(keras.Input(shape=(len(input_labels),)))

model2.add(layers.Dense(units=8, activation="relu"))
model2.add(layers.Dense(units=8, activation="relu"))
# model1.add(layers.Dense(units=16, activation="sigmoid"))

model2.add(layers.Dense(units=len(predict_labels)))

model2.compile(
    optimizer=optimizer,
    loss=cost_function
)








training_history = model1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)


#%% evaluating performance
test_loss = model1.evaluate(X_validate, y_validate)
print(f"Test loss: {test_loss}")





#%% plotting


plt.close('all')
figsize = [12, 8]
fontsize = 18
dpi = 200

pylab.rcParams.update({
    'figure.figsize': figsize,
    'legend.fontsize': fontsize,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'savefig.dpi': dpi
})






# plotting population size
fig = plt.figure(figsize=figsize)
plt.plot(training_history.history["loss"])
plt.title('loss over epochs')
plt.ylim([0, 0.1])
#plt.savefig(r'../report/images/pop2018_hist.png')



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


