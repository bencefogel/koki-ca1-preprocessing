import numpy as np
import pandas as pd


data_dir = 'L:/cluster_seed30/raw_data'
currents = ['AMPA', 'NMDA', 'GABA', 'GABA_B']

def preprocess_synaptic_currents(data_dir, currents):
    dfs = []
    for curr in currents[:1]:
        segments = np.load(data_dir + f'/synaptic_segments/{curr}_segments.npy')
        values = np.load(data_dir + f'/synaptic_currents/{curr}_currents.npy')
        df = pd.DataFrame(data=values, index=segments)
        df = df.reset_index()
        df_summed = df.groupby('index').sum()
        df_summed.columns = df_summed.columns.astype(int)
        df_summed.insert(0, 'itype', curr)
        dfs.append(df_summed)
    return dfs

