import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import keras.layers as kl
import keras.models as km
import keras.callbacks as kc

from allensdk.core.cell_types_cache import CellTypesCache
import matplotlib.pyplot as plt


def resample_timeseries(v, i, t, 
                        hz_in, hz_out):
    factor = int(hz_in / hz_out)

    v = v[::factor]
    i = i[::factor]
    t = t[::factor]
    
    return v, i, t

def load_sweep(ds, sn):
    sweep = ds.get_sweep(sn)
    
    idx0 = sweep['index_range'][0]
    v = sweep['response'] * 1e3 # to mV
    i = sweep['stimulus'] * 1e12 # to pA
    t = np.arange(0, len(v)) * (1.0 / sweep['sampling_rate'])

    v = v[idx0:]    
    i = i[idx0:]
    t = t[idx0:]

    hz_in = sweep['sampling_rate']
    hz_out = 25000.

    return resample_timeseries(v, i, t, hz_in, hz_out)

def load_data(stim_names):
    ctc = CellTypesCache(manifest_file="ctc/manifest.json")
    cells = ctc.get_cells()
    
    cell_id = cells[0]['id']
    sweeps = ctc.get_ephys_sweeps(cell_id)
    sweeps = [ (sweep['sweep_number'],sweep['stimulus_name']) for sweep in sweeps if sweep['stimulus_name'] in stim_names ]

    ds = ctc.get_ephys_data(cell_id)

    vv, ii = [], []

    dur = 2000
    delay = 200

    for sn,st in sweeps:
        v,i,t = load_sweep(ds, sn)        

        idx0 = np.argwhere(i!=0)[0][0] - delay
        
        v = v[idx0:]
        i = i[idx0:]

        if st.startswith('Noise'):
            offs = [ 0, 200000, 400000 ] 
            for off in offs: 
                vv.append(v[off:])
                ii.append(i[off:])
        else:
            vv.append(v)
            ii.append(i)

    stims = np.array([i[:dur] for i in ii])
    resps = np.array([v[:dur] for v in vv]) + 74.0

    print(stims.shape)

    return stims, resps

def train():
    stims, resps = load_data(['Noise 1','Long Square','Short Square'])

    time_steps = stims.shape[1]
    repeats = stims.shape[0]

    model = km.Sequential([
        kl.GRU(20, return_sequences=True, input_shape=(time_steps,1)),
        kl.GRU(20, return_sequences=True),
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
    stims, resps = load_data(['Long Square'])

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
