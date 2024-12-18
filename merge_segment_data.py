import os
import pandas as pd
import numpy as np

# Input directory and files
data_dir = 'L:/cluster_seed30/preprocessed_data/membrane_currents'
index_file = os.path.join(data_dir, 'multiindex.csv')
output_dir = 'L:/cluster_seed30/preprocessed_data/merged_soma'

# Load the index once
index = pd.read_csv(index_file)

# Ensure the index is saved only once
index_saved = False


def merge_soma_segments(index, values):
    """
    Merges soma segments in the dataset.
        Parameters:
            index (df): The multiindex DataFrame containing 'segment' and 'itype' columns.
            values (array): The array of membrane current values corresponding to the multiindex.

        Returns:
            df_updated (df): A DataFrame with soma segments merged and other segments preserved.
    """
    # Create a DataFrame
    multiindex = pd.MultiIndex.from_frame(index)
    df = pd.DataFrame(data=values, index=multiindex)

    # Remove soma segments
    df_soma_removed = df[~df.index.get_level_values(0).str.contains('soma')]

    # Merge soma segments
    df_soma = df[df.index.get_level_values(0).str.contains('soma')].copy()
    df_soma.reset_index(inplace=True)
    df_soma.drop('segment', axis=1, inplace=True)
    df_soma_summed = df_soma.groupby('itype').sum()
    df_soma_summed.reset_index(inplace=True)
    df_soma_summed['segment'] = 'soma'
    df_soma_summed.set_index(['segment', 'itype'], inplace=True)

    # Combine updated data
    df_updated = pd.concat([df_soma_removed, df_soma_summed], axis=0)
    return df_updated


def process_all_files(index, data_dir, output_dir):
    """
        Processes all membrane current value chunks in the directory and saves the results.

        This function loops through all files in the specified directory that match the naming
        convention `current_values_chunk_*.npy`, processes them to merge soma segments, and
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
    for chunk_file in sorted(os.listdir(data_dir)):
        if chunk_file.startswith('current_values_chunk_') and chunk_file.endswith('.npy'):
            chunk_path = os.path.join(data_dir, chunk_file)

            # Load the current values chunk
            values = np.load(chunk_path)

            # Merge soma segments
            df_updated = merge_soma_segments(index, values)

            # Save the updated values chunk
            chunk_number = chunk_file.split('_')[-1].split('.')[0]  # Extract chunk number
            chunk_output_file = os.path.join(output_dir, f"merged_soma_values{chunk_number}.npy")
            np.save(chunk_output_file, df_updated.values)

            # Save the index only once
            if not index_saved:
                index_output_file = os.path.join(output_dir, 'multiindex_merged_soma.csv')
                df_updated.index.to_frame().reset_index(drop=True).to_csv(index_output_file, index=False)
                index_saved = True


# Process all chunks
process_all_files(index, data_dir, output_dir)
