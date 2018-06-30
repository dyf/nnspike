import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import keras.utils as ku

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import argparse

import load_stim_resp as ld
out_size = 440

def train():
    stims, resps = ld.load_data(['Noise 1','Long Square','Short Square'])

    time_steps = stims.shape[1]
    repeats = stims.shape[0]

    model = km.Sequential([
        kl.Conv1D(filters=10, kernel_size=300, padding='causal', input_shape=(None,1), dilation_rate=1),
        kl.Conv1D(filters=10, kernel_size=300, padding='causal', input_shape=(None,1), dilation_rate=2),
        kl.Conv1D(filters=10, kernel_size=300, padding='causal', input_shape=(None,1), dilation_rate=4),       
        kl.Conv1D(filters=10, kernel_size=300, padding='causal', input_shape=(None,1), dilation_rate=8),
    # kl.Conv1D(filters=10, kernel_size=21, activation='relu'),
        kl.Dense(1),
        ])


    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    stims = stims.reshape(stims.shape[0], stims.shape[1], 1)
    resps = resps.reshape(resps.shape[0], resps.shape[1], 1)

    model.fit(stims, resps, epochs=1000, batch_size=100,
              callbacks=[ kc.ModelCheckpoint('output/model_causal_cnn.h5') ])

def vis():
    stims, resps = ld.load_data(['Noise 2'])

    model = km.load_model('output/model_causal_cnn.h5')
    
    output = model.predict(stims.reshape(stims.shape[0], stims.shape[1], 1))
    output.reshape(output.shape[0], output.shape[1])

    import matplotlib.pyplot as plt
    for i in range(output.shape[0]):
        plt.plot(stims[i], label='stim')
        plt.plot(output[i], label='predict')
        plt.plot(resps[i], label='train')
        plt.legend()
        plt.show()
        plt.close()

#train()
vis()