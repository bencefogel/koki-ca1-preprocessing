import os
import pandas as pd
import networkx as nx
from utils import load_df, create_directed_graph


def merge_dendritic_section_iax(df: pd.DataFrame, section: str) -> pd.DataFrame:
    """
   This function selects the axial current connections that are external to the specified dendritic section
   (i.e., connections between parent and children nodes) and merges them back into the dataframe after
   renaming and removing internal connections.

   Parameters:
   ----------
   df : pd.DataFrame
       The dataframe containing axial current connections, with a multi-level index that includes 'ref'
       (reference) and 'par' (parent) segments.
   section : str
       The dendritic section identifier for which external axial current connections are to be merged.

   Returns:
   -------
   pd.DataFrame
       A dataframe with the external axial current connections of the specified dendritic segment merged back
       into the original dataframe, while internal connections are removed.

   Notes:
   ------
   - Internal axial current connections, both as reference and parent, are removed from the dataframe.
   - The function specifically renames certain index values that correspond to internal and section-end-external segments.
   """
    # Select external iax connections (between parent and children nodes)
    df_segment_ref = df[df.index.get_level_values('ref').str.startswith(f'{section}(')]  # select iax rows where segment is the reference
    df_segment_par = df[df.index.get_level_values('par').str.startswith(f'{section}(')]  # select iax rows where segment is the parent
    df_external = pd.concat([df_segment_ref, df_segment_par]).drop_duplicates(keep=False)  # this keeps rows that are unique (meaning that they connect to external nodes)

    # Rename index
    rename_dict = {'dend5_0111111111111111111(0.0454545)': section,  # currently not automatic: first internal and section-end-external segments should be renamed
                   'dend5_0111111111111111111(1)': section}
    df_external.rename(index=rename_dict, level='ref', inplace=True)
    df_external.rename(index=rename_dict, level='par', inplace=True)

    # Remove segment iax rows (both external and internal)
    df_internal_idx = pd.concat([df_segment_ref, df_segment_par]).drop_duplicates().index
    df.drop(df_internal_idx, inplace=True)

    # Concatenate updated external iax rows
    df_merged_dendritic_section = pd.concat([df, df_external])
    return df_merged_dendritic_section

def update_root_node(df_merged: pd.DataFrame, section: str) -> pd.DataFrame:
    """
    Updates the root node in the given dataframe by switching the reference and parent segments along the shortest
    path between a new root and the original root ('soma'), and reversing the axial current (iax) values.

    This function modifies the reference-parent pairs and axial current values of the edges on the shortest path
    between the new root and the original root (soma), updating the dataframe accordingly.

    Parameters:
    ----------
    df_merged : pd.DataFrame
        A dataframe containing axial current (iax) data with a multi-level index consisting of reference ('ref')
        and parent ('par') segments. The new root node should be represented by a section where the segment values
        have already been merged.
    section : str
        The section identifier representing the new root node.

    Returns:
    -------
    pd.DataFrame
        A new dataframe where the axial current connections along the shortest path between the new root and
        the original root ('soma') have been updated by switching the reference-parent pairs and negating the
        axial current values.

    Notes:
    ------
    - The reference and parent segments of the edges on the shortest path are switched, and the axial current values
      are multiplied by -1 to reflect the change in direction.
    - The resulting dataframe is re-indexed and returned, with the reference ('ref') and parent ('par') columns properly set.
    """
    # The input of this function should be a dataframe where the new root node is a section where the segment values are already merged
    dg = create_directed_graph(df_merged, df_merged.columns[0])
    g = dg.to_undirected()

    original_root = 'soma'
    new_root = section

    # Extract iax rows that are on the shortest path between the new root and the soma (original root)
    path = nx.shortest_path(g, source=new_root, target=original_root)  # select nodes of the shortest path between soma and new root
    edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]  # create node pairs for each edge in the path
    df_edges_in_path = df_merged[df_merged.index.isin(edges_in_path)]  # select iax rows of the path

    # Switch ref-par pairs and multiply iax values by -1
    df_switched = df_edges_in_path.copy()
    df_switched.index = pd.MultiIndex.from_tuples([(b, a) for a, b in df_edges_in_path.index])
    df_switched = -df_switched

    # Update original dataframe
    df_updated = df_merged.copy()
    df_updated = df_updated.drop(df_edges_in_path.index)  # drop rows corresponding to the original node pairs in the path
    df_updated = pd.concat([df_updated, df_switched])
    df_updated = df_updated.reset_index()
    df_updated = df_updated.rename(columns={'level_0': 'ref', 'level_1': 'par'})
    df_updated = df_updated.set_index(['ref', 'par'])
    return df_updated

if __name__ == '__main__':
    input_dir = 'E:/cluster_seed30/preprocessed_data/axial_currents_merged_soma'
    index_fname = os.path.join(input_dir, 'multiindex_merged_soma.csv')
    data_fname = os.path.join(input_dir, 'merged_soma_values_0.npy')

    df = load_df(index_fname, data_fname)
    section = 'dend5_0111111111111111111'  # distal apical segment (close to the trunk, not a terminal branch)

    df_merged = merge_dendritic_section_iax(df, section)
    df_updated_root = update_root_node(df_merged, section)