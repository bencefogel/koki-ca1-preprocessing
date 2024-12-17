import numpy as np
import os


def save_in_chunks(current_values, output_dir, chunk_size=None):
    """
    Save the current_values array in chunks along the columns to the specified output directory.

    Parameters:
    - current_values (numpy.ndarray): The array of numerical values to be saved.
    - output_dir (str): The directory where the chunks will be saved.
    - chunk_size (int): The number of columns to save per chunk (default is all columns).
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
