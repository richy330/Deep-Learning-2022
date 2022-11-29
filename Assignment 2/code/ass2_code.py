#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:05:09 2022

@author: richard amering, michael mitterlindner
"""



# required hack because of collision of .dlls on windows, probably due to tensorflow
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import itertools


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

from tensorflow import keras
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from keras import layers
from keras.optimizers import Adam, SGD
from keras.losses import mean_squared_error


#%% setting up hyperparameters
epochs = 50
batch_size = 64
cost_function = mean_squared_error
optimizers = [
    Adam(), 
    SGD(), 
    SGD(momentum=0.5, name='SGD-Mom')
]
activation_functions = [
    'sigmoid', 
    'relu'
]
learning_rates = [
    ExponentialDecay(initial_learning_rate=0.001, decay_steps=20, decay_rate=0.9, name='Exp-dec.'),
    ExponentialDecay(initial_learning_rate=0.0001, decay_steps=20, decay_rate=1, name=f'{0.0001}')
]


#%% adjusting plot appearance
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
    'savefig.dpi': dpi,
    "text.usetex": True,
    "font.family": "serif"
})



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



def create_model(n_units, activation):
    n_inputs, *n_units, n_outputs = n_units
    model = keras.Sequential()
    
    model.add(keras.Input(shape=(n_inputs,)))
    for n_neurons in n_units:
        model.add(layers.Dense(units=n_neurons, activation=activation))
    model.add(layers.Dense(units=n_outputs))
    
    return model







#%% loading and preparing training-, validation- and test-datasets
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
n_neurons = [
    [len(input_labels), 8, 8, len(predict_labels)],
    [len(input_labels), 32, 32, len(predict_labels)],
    [len(input_labels), 32, 8, 32, len(predict_labels)]
]


training_histories = []
validation_losses = []

# we keep track of evaluated hyperparameters in these lists
evaluated_structures = []
evaluated_activations = []
evaluated_optimizers = []
evaluated_learning_rates = []

# we could eliminate this counting variable with the enumerate-function, but don't like
# the combination of enumerate and itertools.product()
n = 0
for n_units, activation, optimizer, learning_rate in itertools.product(n_neurons, activation_functions, optimizers, learning_rates):

    optimizer.learning_rate = learning_rate
    model = create_model(n_units, activation)
    model.compile(
        optimizer=optimizer,
        loss=cost_function
    )
    

    training_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    training_histories.append(training_history)
    
    

    validation_loss = model.evaluate(X_validate, y_validate)
    validation_losses.append(validation_loss)
    print(f"Test loss: {validation_loss}")
    
    # we keep track of hyperparameters in these lists
    evaluated_structures.append(n_units)
    evaluated_activations.append(activation)
    evaluated_optimizers.append(optimizer._name)
    evaluated_learning_rates.append(learning_rate.name)



#%% plotting of training history for each hyperparameter combination
    annotation_text = \
        f"Structure: {n_units}" \
        f"\nActivation: {activation}" \
        f"\nOptimizer: {optimizer._name}" \
        f"\nLearning rate: {learning_rate.name}" \
        f"\nValidation Loss: {validation_loss:.5f}"
        
    fig, ax = plt.subplots()
    ax.plot(training_history.history["loss"])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim([0, None])
    
    ax.text(0.70, 0.95, annotation_text, transform=ax.transAxes, verticalalignment='top', fontsize=fontsize)
    plt.savefig(r'../plots/training_evolution_{}_{}.png'.format(optimizer._name, n))
    plt.show()
    n += 1

    

    


#%% testing the final, best performing model and plotting its training history
n_units_best = [17, 32, 8, 32, 2]
activation_best = 'relu'
learning_rate_best = learning_rates[1]
optimizer_best = Adam(learning_rate=learning_rate_best)

final_model = create_model(n_units_best, activation_best)
final_model.compile(
    optimizer=optimizer_best,
    loss=cost_function
)


X_train_best = pd.concat([X_train, X_validate], axis=0)
y_train_best = pd.concat([y_train, y_validate], axis=0)

training_history = final_model.fit(X_train_best, y_train_best, epochs=epochs, batch_size=batch_size)


validation_loss = model.evaluate(X_test, y_test)
print(f"Test loss: {validation_loss}")


#%% plotting the best performing model
annotation_text = \
    f"Structure: {n_units_best}" \
    f"\nActivation: {activation_best}" \
    f"\nOptimizer: {optimizer_best._name}" \
    f"\nLearning rate: {learning_rate_best.name}" \
    f"\nValidation Loss: {validation_loss:.5f}"
    
fig, ax = plt.subplots()
ax.plot(training_history.history["loss"])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim([0, None])
plt.grid()

ax.text(0.70, 0.95, annotation_text, transform=ax.transAxes, verticalalignment='top', fontsize=fontsize)
plt.savefig(r'../plots/training_evolution_{}.png'.format('best_model'))
plt.show()


