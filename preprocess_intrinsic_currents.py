import numpy as np
import pandas as pd
from tqdm import tqdm


def change_unit_na(currents: pd.DataFrame, area: pd.DataFrame) -> pd.DataFrame:
    """
    Convert membrane currents to nA from mA/cm2.

    Parameters:
        currents (df): DataFrame containing membrane currents.
        area (df): DataFrame containing segment areas.

    Returns
        df_converted (df): DataFrame containing membrane currents in nA.
    """
    array_converted = np.zeros_like(currents.values)
    for i, segment in enumerate(currents.index):
        segment_area = area.loc[segment].values[0]
        array_na = currents.loc[segment] * segment_area * 0.01
        array_converted[i, :] = array_na

    df_converted = pd.DataFrame(data=array_converted, index=list(currents.index), columns=list(currents.columns))
    df_converted = df_converted.reset_index()
    return df_converted


def preprocess_intrinsic_currents(data_dir, currents, area):
    """
    Preprocess intrinsic current data by converting units, and organizing it into dataframes.

    Parameters:
        data_dir (str):
            The directory path where the raw intrinsic current data is stored.
            This directory must contain 'intrinsic_segments' and 'intrinsic_currents' subdirectories.

        currents (list of str):
            A list of intrinsic current types to process.

        area (df):
            A DataFrame containing segment area information, which is used to convert the raw current values.

    Returns:
        dfs (list of df):
            A list of DataFrames, where each DataFrame corresponds to a processed intrinsic current type.
            Each DataFrame has the following structure:
                - `index` : segment index (as a category)
                - `itype` : intrinsic current type (as a category)
                - Remaining columns : current values, converted to nA.

    Notes:
    - The function reads `.npy` files for segment indices and corresponding current values.
    - Columns 'index' and 'itype' are optimized by converting them to categorical data types for memory efficiency.
    """
    dfs = []
    for curr in tqdm(currents):
        segments = np.load(data_dir + f'/intrinsic_segments/{curr}_segments.npy').astype(str)
        values = np.load(data_dir + f'/intrinsic_currents/{curr}_currents.npy').astype(np.float32)
        df = pd.DataFrame(data=values, index=segments)
        df_converted = change_unit_na(df, area)
        df_converted.insert(1, 'itype', curr)
        df_converted[['index', 'itype']] = df_converted[['index', 'itype']].astype('category')
        dfs.append(df_converted)
    return dfs
