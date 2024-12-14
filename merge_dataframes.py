import pandas as pd
import numpy as np
import os

from preprocess_intrinsic_currents import preprocess_intrinsic_currents
from preprocess_synaptic_currents import preprocess_synaptic_currents


data_dir = 'L:/cluster_seed30/raw_data'
intrinsic_currents = ['nax', 'nad', 'kap', 'kad', 'kdr', 'kslow', 'car', 'passive', 'capacitive']
synaptic_currents = ['AMPA', 'NMDA', 'GABA', 'GABA_B']
segment_area = pd.read_csv(data_dir + '/segment_area.csv', index_col=0)

dfs_intrinsic = preprocess_intrinsic_currents(data_dir, intrinsic_currents, segment_area)
dfs_synaptic = preprocess_synaptic_currents(data_dir, synaptic_currents)

dfs = dfs_intrinsic + dfs_synaptic
df_im = pd.concat(dfs)
df_im = df_im.reset_index()
segments = df_im['index'].unique()
itypes = df_im['itype'].unique()

multi_index = pd.MultiIndex.from_product([segments, itypes], names=['segment', 'itype'])
df_im_combined = df_im.set_index(['index', 'itype']).reindex(multi_index)
df_im_combined = df_im_combined.fillna(0)
df_im_combined.columns = df_im_combined.columns.astype(int)

index_df = pd.DataFrame(df_im_combined.index.tolist(), columns=['segment', 'itype'])
output_dir = "L:/cluster_seed30/preprocessed_data"
os.makedirs(output_dir, exist_ok=True)
index_file = os.path.join(output_dir, "multiindex.csv")
index_df.to_csv(index_file, index=False)

current_values = df_im_combined.values.astype(np.float32)
output_dir = "L:/cluster_seed30/preprocessed_data"
os.makedirs(output_dir, exist_ok=True)
numerical_file = f"{output_dir}/current_values_32.npy"
np.save(numerical_file, current_values)
print(f"Numerical values saved as NumPy array to {numerical_file}")
