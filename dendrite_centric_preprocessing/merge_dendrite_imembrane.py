import os
import pandas as pd

from dendrite_centric_preprocessing.merge_dendrite_iax import df_merged_dendritic_segment
from utils import load_df

def merge_dendritic_section_imembrane(df: pd.DataFrame, section: str) -> pd.DataFrame:
    """
    Merges data for a specific dendritic section, summing  the values for each `itype` across the segments of the section

    Parameters:
    ----------
    df : pd.DataFrame
        A pandas DataFrame where rows are indexed by a multi-level index. The first level of the index is a segment
        identifier, and the second level represents `itype`.

    section : str
        The name of the dendritic section to be processed. This will be used to select the rows that belong to the
        given section.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame that combines the original data excluding the selected dendritic section and the summed data
        for that section grouped by `itype`. The new DataFrame has the dendritic segment and `itype` as a two-level index.
    """
    df_dend = df[df.index.get_level_values(0).str.startswith(f'{section}(')]  # select all rows belonging to the given segment
    df_summed_by_itype = df_dend.groupby(level='itype').sum()  # sum dataframe by current type for each time point
    df_summed_by_itype = df_summed_by_itype.reset_index()
    df_summed_by_itype['segment'] = section
    df_summed_by_itype = df_summed_by_itype.set_index(['segment', 'itype'])

    # Update original dataframe with the merged dendritic segment
    df_merged_dendritic_segment = pd.concat([df.drop(df_dend.index), df_summed_by_itype], axis=0)
    return df_merged_dendritic_segment

if __name__ == '__main__':
    input_dir = 'E:/cluster_seed30/preprocessed_data/membrane_currents_merged_soma'
    index_fname = os.path.join(input_dir, 'multiindex_merged_soma.csv')
    data_fname = os.path.join(input_dir, 'merged_soma_values_0.npy')

    df = load_df(index_fname, data_fname)
    segment = 'dend5_0111111111111111111'  # distal apical segment (close to the trunk, not a terminal branch)

    df_merged_dendritic_section = merge_dendritic_section_imembrane(df, segment)
