import numpy as np
import random

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
import h5py


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

    cats = []
    patches = []
    for arr, cat in zip((si, pi, ti, noev), (0, 1, 2, 3)):
        if len(arr) == 0:
            continue

        idxs = np.random.choice(arr, N)
        for idx in idxs:
            r = idx-hp, idx+hp
            if r[0] > 0 and r[1] <= len(v):
                cats.append(cat)
                patches.append(v[r[0]:r[1]])
    
    return np.array(cats), np.vstack(patches)

def sample_data_set(ds, N, N_sweep, patch_size):
    sweep_nums = ds.get_sweep_numbers()
    
    ct = 0
    for i in range(1000):
        if ct >= N:
            break

        idx = random.randint(0,len(sweep_nums)-1)
        sweep_num = sweep_nums[idx]
        
        v, i, t, si, pi, ti = load_sweep(ds, sweep_num)

        cats, patches = grab_patches(v, i, t,
                                     si, pi, ti,
                                     N_sweep, patch_size)

        if len(cats) > 0:
            yield cats, patches

        ct += cats.shape[0]
        print(ct)

def sample_data_sets(cells, ctc, num_cells, patches_per_cell, patches_per_grab, patch_size):
    idxs = np.random.choice(np.arange(len(cells)), num_cells)
    for idx in idxs:
        cell = cells[idx]
        ds = ctc.get_ephys_data(cell['id'])

        for cats, patches in sample_data_set(ds, patches_per_cell, patches_per_grab, patch_size):
            yield cats, patches

from allensdk.config import enable_console_log
enable_console_log()

ctc = CellTypesCache()
cells = ctc.get_cells()

for i, (cats, patches) in enumerate(sample_data_sets(cells, ctc, 100, 1000, 100, 2000)):
    print(i)
    np.save('patches/cats_%04d.npy' % i, cats)
    np.save('patches/patches_%04d.npy' % i, patches)

