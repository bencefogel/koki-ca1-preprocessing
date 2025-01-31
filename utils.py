import numpy as np
import pandas as pd
import networkx as nx
import os


def save_in_chunks(current_values, output_dir, chunk_size=None):
    """
    Save the current_values array in chunks along the columns to the specified output directory.

    Parameters:
        current_values (numpy.ndarray): The array of numerical values to be saved.
        output_dir (str): The directory where the chunks will be saved.
        chunk_size (int): The number of columns to save per chunk (default is all columns).
    """
    os.makedirs(output_dir, exist_ok=True)

    # If no chunk_size is provided, save the whole array in one file
    if chunk_size is None:
        chunk_size = current_values.shape[1]

    num_chunks = current_values.shape[1] // chunk_size + (1 if current_values.shape[1] % chunk_size != 0 else 0)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, current_values.shape[1])

        chunk_values = current_values[:, start_idx:end_idx]

        chunk_file = os.path.join(output_dir, f"current_values_chunk_{i}.npy")

        np.save(chunk_file, chunk_values)
        print(f"Saved column chunk {i} to {chunk_file}")


def load_df(index_fname: str, values_fname: str):
    """
    Loads a DataFrame from a CSV file containing a multiindex and a NumPy file containing the corresponding values.

    Parameters:
        index_fname (str): The file path to the CSV file containing the multiindex data.
        values_fname (str): The file path to the .npy file containing the array of values.

    Returns:
        pd.DataFrame: A pandas DataFrame constructed using the multiindex from the CSV file and the values from the .npy file.
    """
    index = pd.read_csv(index_fname)
    values = np.load(values_fname)

    multiindex = pd.MultiIndex.from_frame(index)
    df = pd.DataFrame(data=values, index=multiindex)
    return df


def get_iax(df_iax, segment):
    """
    Extracts and concatenates the axial currents (`iax`) for a given dendritic segment from both 'ref'
    and 'par' levels of the index.

    Parameters:
    ----------
    df_iax : pd.DataFrame
        A DataFrame indexed by multi-level indexes, where 'ref' and 'par' are levels of the index representing
        the reference and the parent segments, and the values represent the calculated axial currents.

    segment : str
        The segment name to filter and extract iax for.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the axial currents for the given segment.
    """
    ref_mask = df_iax.index.get_level_values("ref") == segment
    ref_iax = -1 * df_iax[ref_mask]

    par_mask = df_iax.index.get_level_values("par") == segment
    par_iax = df_iax[par_mask]

    df_iax_seg = pd.concat([ref_iax, par_iax], axis=0)
    return df_iax_seg


def create_directed_graph(df_iax: pd.DataFrame, tp: int) -> nx.DiGraph:
    """
   Creates a directed graph based on the axial current data (`iax`) at a specific timepoint.
   The graph is constructed using the 'ref' and 'par' columns of the DataFrame as nodes,
   and directed edges are added between these nodes based on the sign of `iax`.

   Parameters:
   ----------
   df_iax : pd.DataFrame
        A DataFrame indexed by multi-level indexes, where 'ref' and 'par' are levels of the index representing
        the reference and the parent segments, and the values represent the calculated axial currents.

   tp : int
       The time point index to filter the iax values from the DataFrame.

   Returns:
   --------
   nx.DiGraph
       A directed graph where nodes represent segments,
       and directed edges are created based on the sign of `iax` at the specified time point.
   """
    df_iax_tp = df_iax[tp]
    df_iax_tp = df_iax_tp.reset_index()
    df_iax_tp.rename(columns={tp: "iax"}, inplace=True)  # has three columns: ref, par, iax

    # Create directed graph (add edges to the graph based on the sign of iax)
    dg = nx.DiGraph()
    for index, row in df_iax_tp.iterrows():
        if row['iax'] >= 0:
            dg.add_edge(row['par'], row['ref'], iax=row['iax'])  # par -> ref if 'iax_timepoint' is positive
        elif row['iax'] < 0:
            dg.add_edge(row['ref'], row['par'], iax=row['iax'])  # ref -> par if 'iax_timepoint' is negative
    return dg