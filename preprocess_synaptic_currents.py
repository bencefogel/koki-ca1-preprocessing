import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess_synaptic_currents(data_dir, currents):
    dfs = []
    for curr in tqdm(currents):
        segments = np.load(data_dir + f'/synaptic_segments/{curr}_segments.npy').astype(str)
        values = np.load(data_dir + f'/synaptic_currents/{curr}_currents.npy').astype(np.float32)
        df = pd.DataFrame(data=values, index=segments)
        df = df.reset_index()
        df_summed = df.groupby('index', as_index=False).sum()
        df_summed.insert(1, 'itype', curr)
        df_summed[['index', 'itype']] = df_summed[['index', 'itype']].astype('category')
        dfs.append(df_summed)
    return dfs
