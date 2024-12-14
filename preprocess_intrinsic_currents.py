import numpy as np
import pandas as pd


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

    df_converted = pd.DataFrame(data=array_converted, index=currents.index, columns=currents.columns)
    return df_converted


def preprocess_intrinsic_currents(data_dir, currents, segment_area):
    dfs = []
    for curr in currents[:1]:
        segments = np.load(data_dir + f'/intrinsic_segments/{curr}_segments.npy')
        values = np.load(data_dir + f'/intrinsic_currents/{curr}_currents.npy')
        df = pd.DataFrame(data=values, index=segments)
        df_converted = change_unit_na(df, segment_area)
        df_converted.columns = list(df_converted.columns)
        df_converted.insert(0, 'itype', curr)
        dfs.append(df_converted)
    return dfs




