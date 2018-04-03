import numpy as np

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor


def load_sweep(ds, sweep_num):
    sweep = ds.get_sweep(sweep_num)
    v = sweep['response'] * 1e3 # to mV
    i = sweep['stimulus'] * 1e12 # to pA
    t = np.arange(0, len(v)) * (1.0 / sweep['sampling_rate'])

    sweep_ext = EphysSweepFeatureExtractor(t=t, v=v, i=i)
    sweep_ext.process_spikes()
    
    spike_idxs = sweep_ext.spike_feature("threshold_index")
    peak_idxs = sweep_ext.spike_feature("peak_index")
    trough_idxs = sweep_ext.spike_feature("trough_index")

    hz_in = sweep['sampling_rate']
    hz_out = 10000.

    return resample_timeseries(v, i, t, 
                               spike_idxs, peak_idxs, trough_idxs,
                               hz_in, hz_out)

def resample_timeseries(v, i, t, 
                        si, pi, ti,
                        hz_in, hz_out):
    factor = int(hz_in / hz_out)

    v = v[::factor]
    i = i[::factor]
    t = t[::factor]

    si = (si.astype(float) / factor).astype(int)
    pi = (si.astype(float) / factor).astype(int)
    ti = (si.astype(float) / factor).astype(int)
    
    return v, i, t, si, pi, ti

def grab_patches(v, i, t, si, pi, ti, N, patch_size):
    hp = patch_size // 2

    noev = list(set(np.arange(len(v))) - set(si.tolist()) - set(pi.tolist()) - set(ti.tolist()))

    patches = []
    for arr, cat in zip((si, pi, ti, noev), (0, 1, 2, 3)):
        idxs = np.random.choice(arr, N)
        for idx in idxs:
            r = idx-hp, idx+hp
            patches.append((cat, v[r[0]:r[1]]))
    
    return patches
    
patch_size = 2000
cell_id = 507121819
sweep_num = 35

ctc = CellTypesCache()
cells = ctc.get_cells()

ds = ctc.get_ephys_data(cell_id)
v, i, t, si, pi, ti = load_sweep(ds, sweep_num)

patches = grab_patches(v, i, t,
                       si, pi, ti,
                       10, patch_size)

print(patches)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.plot(patches[0][1])
plt.savefig('test.png')

    
