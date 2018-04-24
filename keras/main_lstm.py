import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import keras.layers as kl
import keras.models as km
import keras.callbacks as kc

from allensdk.core.cell_types_cache import CellTypesCache

def load_data():
    ctc = CellTypesCache(manifest_file="ctc/manifest.json")
    cells = ctc.get_cells()
    
    cell_id = cells[0]['id']
    sweeps = ctc.get_ephys_sweeps(cell_id)
    noise_sweeps = [ sweep['sweep_number'] for sweep in sweeps if sweep['stimulus_name'] == 'Noise 1' ]

    ds = ctc.get_ephys_data(cell_id)
    data = ds.get_sweep(noise_sweeps[0])

    return data

data = load_data()
time_steps = len(data['response'])

model = km.Sequential([
    kl.LSTM(10, return_sequences=True, input_shape=(time_steps,1)),
    kl.LSTM(10, return_sequences=True),
    kl.Dense(1)
    ])

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

stim = data['stimulus'].reshape(1, time_steps, 1)
resp = data['response'].reshape(1, time_steps, 1)

model.fit(stim, resp, epochs=10, batch_size=1,
          callbacks=[ kc.ModelCheckpoint('output/model_lstm.h5') ])


