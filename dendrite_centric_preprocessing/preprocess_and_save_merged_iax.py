import os
import numpy as np
from tqdm import tqdm

from merge_dendrite_iax import merge_dendritic_section_iax, update_root_node
from utils import load_df


# Input and output parameters
data_dir = 'E:/cluster_seed30/preprocessed_data/axial_currents_merged_soma'
index_file_path = os.path.join(data_dir, 'multiindex_merged_soma.csv')
output_dir = 'E:/cluster_seed30/preprocessed_data/dendrite_centric/axial_currents_merged_dendrite'
section = 'dend5_0111111111111111111'

# Ensure the index is saved only once
index_saved = False

def process_all_files(index_file_path, data_dir, output_dir):
    """
    Processes all axial current chunks in the directory and saves the results.
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
            df_merged = merge_dendritic_section_iax(df, section)
            df_updated_root = update_root_node(df_merged, section)

            # Save the updated values chunk
            chunk_number = chunk_file.split('_')[-1].split('.')[0]  # Extract chunk number
            chunk_output_file = os.path.join(output_dir, f"merged_dendrite_values_{chunk_number}.npy")
            np.save(chunk_output_file, df_updated_root.values)

            # Save the index only once
            if not index_saved:
                index_output_file = os.path.join(output_dir, 'multiindex_merged_dendrite.csv')
                df_updated_root.index.to_frame().reset_index(drop=True).to_csv(index_output_file, index=False)
                index_saved = True

process_all_files(index_file_path, data_dir, output_dir)