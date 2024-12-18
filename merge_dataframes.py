import time

import pandas as pd
import numpy as np
import os
import gc

from preprocess_intrinsic_currents import preprocess_intrinsic_currents
from preprocess_synaptic_currents import preprocess_synaptic_currents
from utils import save_in_chunks


input_dir = 'L:/cluster_seed30/raw_data'
intrinsic_currents = ['nax', 'nad', 'kap', 'kad', 'kdr', 'kslow', 'car', 'passive', 'capacitive']
synaptic_currents = ['AMPA', 'NMDA', 'GABA', 'GABA_B']
segment_area = pd.read_csv(input_dir + '/segment_area.csv', index_col=0)

dfs_intrinsic = preprocess_intrinsic_currents(input_dir, intrinsic_currents, segment_area)
dfs_synaptic = preprocess_synaptic_currents(input_dir, synaptic_currents)

# Create merged dataframe
dfs = dfs_intrinsic + dfs_synaptic
del dfs_intrinsic, dfs_synaptic
gc.collect()

df_im = pd.concat(dfs)
del dfs
gc.collect()

df_im['index'] = df_im['index'].astype('category')
df_im['itype'] = df_im['itype'].astype('category')

# Calculate and set multiindex
segments = df_im['index'].unique()
itypes = df_im['itype'].unique()

multi_index = pd.MultiIndex.from_product([segments, itypes], names=['segment', 'itype'])
df_im_combined = df_im.set_index(['index', 'itype']).reindex(multi_index)
del df_im
gc.collect()

df_im_combined = df_im_combined.fillna(0)
df_im_combined.columns = df_im_combined.columns.astype(int)

output_dir = "L:/cluster_seed30/preprocessed_data/membrane_currents"
# Save multiindex as a dataframe
index_df = pd.DataFrame(df_im_combined.index.tolist(), columns=['segment', 'itype'])
os.makedirs(output_dir, exist_ok=True)
index_file = os.path.join(output_dir, "multiindex.csv")
index_df.to_csv(index_file, index=False)

# Save current values as arrays
current_values = df_im_combined.values
save_in_chunks(current_values, output_dir, chunk_size=20000)
