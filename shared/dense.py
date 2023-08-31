import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import sys
import matplotlib.pyplot as plt



# def MLP(nvars,NTRIALS=10):
#     inputs = Input((nvars, ))
#     inputs_trials = tf.expand_dims(inputs,1)
#     inputs_trials = tf.tile(inputs_trials,[1,NTRIALS,1])
    
#     layer = Conv1D(50, kernel_size = 1, strides=1,
#                    activation='relu')(inputs_trials)
    
#     layer = Conv1D(nvars, kernel_size = 1, strides=1,
#                    activation='relu')(layer)

#     layer = Conv1D(50, kernel_size = 1, strides=1,
#                    activation='relu')(layer+inputs_trials)
    
#     layer = Conv1D(1, kernel_size = 1, strides=1,
#                    activation=None)(layer)
    
#     outputs = tf.reduce_mean(layer,1) #Average over trials
#     outputs =tf.keras.activations.sigmoid(outputs)

#     return inputs,outputs

# Records the weights throughout the training process
weights_history = []

# A custom callback
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
class MyCallback(tf.keras.callbacks.Callback):

    def on_batch_end(self, batch, logs):
        weights, _biases = model.get_weights()
        w = weights
        weights = [w1[0]]
        print('on_batch_end() model.weights:', weights)
        weights_history.append(weights)

# callback = MyCallback()

# plt.figure(1, figsize=(6, 3))
# plt.plot(weights_history)
# plt.show()


def MLP(nvars,NTRIALS=10):
    inputs = Input((nvars, ))
    net_trials = []
    for _ in range(NTRIALS):            
        layer = Dense(64,activation='relu')(inputs)
        layer = Dense(128, activation='relu')(layer)
        layer = Dense(64,activation='relu')(layer)
        #layer = Dropout(0.05)(layer) 
        layer = Dense(1,activation='sigmoid')(layer)
        # do not do this - will arbitrarily bias the output NN
        # layer = tf.clip_by_value(layer, clip_value_min=0, clip_value_max=10)
        net_trials.append(layer)

    # variance = tfp.stats.variance(net_trials,0)
    # variance = tf.math.reduce_variance(net_trials,0)
    # variance = tfp.stats.percentile(net_trials, [10,90], interpolation='midpoint',axis=0)

    outputs = tf.reduce_mean(net_trials,0) #Average over trials
    #outputs = tfp.stats.percentile(net_trials, 50.0, interpolation='midpoint', axis=0) # median less sensitive to outliers
    return inputs,outputs
