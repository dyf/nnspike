import numpy as np
import random
import re
import glob
import h5py

random.seed(0)
np.random.seed(0)

from allensdk.config import enable_console_log
enable_console_log()
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
    hz_out = 25000.

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
    pi = (pi.astype(float) / factor).astype(int)
    ti = (ti.astype(float) / factor).astype(int)
    
    return v, i, t, si, pi, ti

def grab_patches(v, i, t, si, pi, ti, N, patch_size):
    c = np.zeros((3, v.shape[0]), dtype=np.uint8)
    c[0,si] = 1
    c[1,pi] = 1
    c[2,ti] = 1
    
    hp = patch_size // 2

    cats = []
    patches = []
    for arr, cat in zip((si, pi, ti), (0, 1, 2)):
        if len(arr) == 0:
            continue
        
        idxs = np.random.choice(arr, N)
        for idx in idxs:
            idx = idx + random.randint(-hp,hp)
            r = idx-hp, idx+hp
            if r[0] > 0 and r[1] <= len(v):
                pv = v[r[0]:r[1]]
                cv = c[:,r[0]:r[1]]

                patches.append(pv)
                cats.append(cv)
    if len(patches) == 0:
        return [], []
    
    return np.stack(cats, axis=0), np.vstack(patches)

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

        n_patches = len(cats)
        if n_patches > 0:
            yield cats, patches

        ct += n_patches
        print(ct)

def sample_data_sets(cells, ctc, num_cells, patches_per_cell, patches_per_grab, patch_size):
    idxs = np.random.choice(np.arange(len(cells)), num_cells)
    for idx in idxs:
        cell = cells[idx]
        ds = ctc.get_ephys_data(cell['id'])

        yield from sample_data_set(ds, patches_per_cell, patches_per_grab, patch_size)

def download():
    ctc = CellTypesCache(manifest_file='ctc/manifest.json')
    cells = ctc.get_cells()

    for i, (cats, patches) in enumerate(sample_data_sets(cells, ctc, 100, 1000, 100, 200)):
        print(cats.shape)
        np.save('patches/cats_%04d.npy' % i, cats)
        np.save('patches/patches_%04d.npy' % i, patches)

def combine():
    patches = []
    cats = []

    pat = re.compile('.*?_(\d+).npy')
    for f in sorted(glob.glob('patches/patches*')):
        m = re.match(pat, f)
        if m:
            print(f)
            pid = int(m.group(1))
            pfile = 'patches/patches_%04d.npy' % pid
            cfile = 'patches/cats_%04d.npy' % pid
            
            patches.append(np.load(pfile))
            cats.append(np.load(cfile))

            
    patches = np.concatenate(patches)
    cats = np.concatenate(cats)

    mean = patches.mean()
    std = patches.std()
    patches = (patches - mean) / std

    print(patches.shape)
    print(cats.shape)

    with h5py.File('training_data.h5', 'w') as f:
        f.create_dataset('patches', data=patches)
        f.create_dataset('cats', data=cats)

download()
combine()
