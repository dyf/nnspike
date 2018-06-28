import numpy as np

from allensdk.core.cell_types_cache import CellTypesCache

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

def load_data(stim_names, reps=10, dur=3000, delay=200):
    ctc = CellTypesCache(manifest_file="ctc/manifest.json")
    cells = ctc.get_cells()
    
    cell_id = cells[0]['id']
    sweeps = ctc.get_ephys_sweeps(cell_id)
    sweeps = [ (sweep['sweep_number'],sweep['stimulus_name']) for sweep in sweeps if sweep['stimulus_name'] in stim_names ]

    ds = ctc.get_ephys_data(cell_id)

    vv, ii = [], []

    for sn,st in sweeps:
        v,i,t = load_sweep(ds, sn)        

        stim_start = np.argwhere(i!=0)[0][0]

        for rep in range(reps):
            idx0 = stim_start - delay - np.random.randint(0, dur//2)
                        
            vr = v[idx0:]
            ir = i[idx0:]

            if st.startswith('Noise'):
                offs = [ 0, 200000, 400000 ] 
                for off in offs: 
                    vv.append(vr[off:off+dur])
                    ii.append(ir[off:off+dur])
            else:
                vv.append(vr[:dur])
                ii.append(ir[:dur])

    stims = np.vstack(ii)
    resps = np.vstack(vv) + 74.0

    print(stims.shape)

    return stims, resps