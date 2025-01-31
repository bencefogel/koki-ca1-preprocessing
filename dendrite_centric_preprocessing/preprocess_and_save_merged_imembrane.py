import os
import numpy as np
from tqdm import tqdm

from merge_dendrite_imembrane import merge_dendritic_section_imembrane
from utils import load_df


# Input and output parameters
data_dir = 'E:/cluster_seed30/preprocessed_data/membrane_currents_merged_soma'
index_file_path = os.path.join(data_dir, 'multiindex_merged_soma.csv')
output_dir = 'E:/cluster_seed30/preprocessed_data/dendrite_centric/membrane_currents_merged_dendrite'
section = 'dend5_0111111111111111111'

# Ensure the index is saved only once
index_saved = False

def process_all_files(index_file_path, data_dir, output_dir):
    """
    Processes all membrane current value chunks in the directory and saves the results.

    This function loops through all files in the specified directory that match the naming
    convention `current_values_chunk_*.npy`, processes them to merge dendritic segments, and
    saves the resulting values and index.

    Parameters:
        index (df): The multiindex DataFrame containing 'segment' and 'itype' columns.
        data_dir (str): The directory containing `multiindex.csv` and the `.npy` value chunks.
        output_dir (str): The directory where the processed files will be saved.

    Returns:
        None: Saves the processed chunks and the merged index to the output directory.
        """
    global index_saved

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all chunk files in the data directory
    for chunk_file in tqdm(sorted(os.listdir(data_dir))):
        if chunk_file.startswith('merged_soma_values_') and chunk_file.endswith('.npy'):
            chunk_path = os.path.join(data_dir, chunk_file)

            # Merge dendritic segments
            df = load_df(index_file_path, chunk_path)
            df_merged_dendritic_segments = merge_dendritic_section_imembrane(df, section)

            # Save the updated values chunk
            chunk_number = chunk_file.split('_')[-1].split('.')[0]  # Extract chunk number
            chunk_output_file = os.path.join(output_dir, f"merged_dendrite_values_{chunk_number}.npy")
            np.save(chunk_output_file, df_merged_dendritic_segments.values)

            # Save the index only once
            if not index_saved:
                index_output_file = os.path.join(output_dir, 'multiindex_merged_dendrite.csv')
                df_merged_dendritic_segments.index.to_frame().reset_index(drop=True).to_csv(index_output_file, index=False)
                index_saved = True

process_all_files(index_file_path, data_dir, output_dir)
