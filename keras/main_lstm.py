import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import keras.layers as kl
import keras.models as km
import keras.callbacks as kc

from allensdk.core.cell_types_cache import CellTypesCache

def resample_timeseries(v, i, t, 
                        hz_in, hz_out):
    factor = int(hz_in / hz_out)

    v = v[::factor]
    i = i[::factor]
    t = t[::factor]
    
    return v, i, t

def load_data():
    ctc = CellTypesCache(manifest_file="ctc/manifest.json")
    cells = ctc.get_cells()
    
    cell_id = cells[0]['id']
    sweeps = ctc.get_ephys_sweeps(cell_id)
    noise_sweeps = [ sweep['sweep_number'] for sweep in sweeps if sweep['stimulus_name'] == 'Noise 1' ]

    ds = ctc.get_ephys_data(cell_id)

    vv, ii = [], []

    for sn in noise_sweeps:
        sweep = ds.get_sweep(sn)
        v = sweep['response'] * 1e3 # to mV
        i = sweep['stimulus'] * 1e12 # to pA
        t = np.arange(0, len(v)) * (1.0 / sweep['sampling_rate'])
        hz_in = sweep['sampling_rate']
        hz_out = 25000.

        v,i,t = resample_timeseries(v, i, t, hz_in, hz_out)

        vv.append(v)
        ii.append(i)

    return np.array(ii), np.array(vv)

stims, resps = load_data()

ts = [ [80000, 100000], [280000, 300000], [480000, 500000] ]
stims = np.vstack([ stims[:,t[0]:t[1]] for t in ts ])
resps = np.vstack([ resps[:,t[0]:t[1]] for t in ts ])

print(stims.shape)

time_steps = stims.shape[1]
repeats = stims.shape[0]

model = km.Sequential([
    kl.LSTM(10, return_sequences=True, input_shape=(time_steps,1)),
    kl.LSTM(10, return_sequences=True),
    kl.Dense(1)
    ])

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

stims = stims.reshape(stims.shape[0], stims.shape[1], 1)
resps = resps.reshape(resps.shape[0], resps.shape[1], 1)

model.fit(stims, resps, epochs=10, batch_size=1,
          callbacks=[ kc.ModelCheckpoint('output/model_lstm.h5') ])


