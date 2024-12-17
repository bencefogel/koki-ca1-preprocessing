import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess_synaptic_currents(data_dir, currents):
    """
        Preprocess synaptic current data by summing over segments, and organizing it into DataFrames.
        Parameters:
            data_dir (str):
                The directory path where the raw synaptic current data is stored.
                This directory must contain 'synaptic_segments' and 'synaptic_currents' subdirectories.

            currents (list of str):
                A list of synaptic current types to process.

        Returns:
            dfs (list of df):
                A list of DataFrames, where each DataFrame corresponds to a processed synaptic current type.
                Each DataFrame has the following structure:
                    - `index` : segment index (as a category)
                    - `itype` : synaptic current type (as a category)
                    - Remaining columns : summed current values.

        Notes:
        - The data is grouped and summed over unique segments using the `groupby` method.
        - Columns 'index' and 'itype' are converted to categorical data types to optimize memory usage.
        """
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
