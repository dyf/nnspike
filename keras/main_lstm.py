import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import keras.layers as kl
import keras.models as km
import keras.callbacks as kc

import matplotlib.pyplot as plt
import load_stim_resp as ld

np.random.seed(0)

def train():
    stims, resps = ld.load_data(['Noise 1','Long Square','Short Square'])

    time_steps = stims.shape[1]
    repeats = stims.shape[0]

    model = km.Sequential([
        kl.GRU(50, return_sequences=True, input_shape=(time_steps,1)),
        kl.GRU(50, return_sequences=True),
        kl.GRU(50, return_sequences=True),
        kl.GRU(50, return_sequences=True),
        kl.GRU(50, return_sequences=True),
        kl.GRU(50, return_sequences=True),
        kl.GRU(50, return_sequences=True),
        kl.GRU(50, return_sequences=True),
        kl.Dense(1)
        ])

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['accuracy'])

    stims = stims.reshape(stims.shape[0], stims.shape[1], 1)
    resps = resps.reshape(resps.shape[0], resps.shape[1], 1)

    model.fit(stims, resps, epochs=1000, batch_size=10,
            callbacks=[ kc.ModelCheckpoint('output/model_lstm.h5') ])

def vis():
    stims, resps = load_data(['Noise 2'])

    model = km.load_model('output/model_lstm.h5')
    
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
    
train()
#vis()
