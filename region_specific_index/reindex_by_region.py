import pandas as pd
import os


def create_region_specific_index(df: pd.DataFrame, input_dir: str) -> pd.DataFrame:
    """
    Creates region-specific index by mapping each segment to a predefined region
    and categorizing intrinsic and synaptic types.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the original multi-indexed data with 'segment' and 'itype' columns

    input_dir : str
        The directory containing text files corresponding to different regions.
        Each file should have a list of segment names associated with that region.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with:
        - 'segment': The original segment names.
        - 'itype': A combined label of the mapped region and current type (e.g., 'axon_intrinsic').

    Notes:
    ------
    - The function reads predefined region text files and maps segments to their respective regions.
    - If a segment is not found in any region list, it is labeled as 'Unknown'.
    - The function also categorizes current types as either 'intrinsic' or 'synaptic'.
    - The final 'itype' column is a combination of the detected region and type.
    """
    fnames_regions = ['distal', 'oblique_trunk', 'axon', 'basal', 'soma']

    # Create a dictionary with each region's file contents split by newline
    regions_dict = {}
    for f in fnames_regions:
        with open(os.path.join(input_dir, f + '.txt'), 'r') as file:
            contents = file.read().strip()  # Read the file and strip leading/trailing whitespace
            segments = contents.split('\n')  # Split the contents by newline
            key = os.path.splitext(f)[0]  # Use the file name (without extension) as the key
            regions_dict[key] = segments  # Store the list of segments as the value

    region_values = []
    segments = df['segment'].values
    for segment in segments:
        seg = segment.split('(')[0].strip()  # Clean the segment
        region_value = 'Unknown'
        # Check if the segment exists in any of the lists in regions_dict
        for key, value in regions_dict.items():
            if seg in value:  # Check if the cleaned segment exists in the list of segments
                region_value = key  # Assign the corresponding region key
                break
        region_values.append(region_value)

    # Create a dictionary that categorizes current types
    type_dict = {'intrinsic':['capacitive', 'car', 'kad', 'kap', 'kdr', 'kslow', 'nad', 'nax', 'passive'],
                 'synaptic':['AMPA', 'GABA', 'GABA_B', 'NMDA']}

    type_values = []
    types = df['itype'].values
    for type in types:
        type_value = 'Unknown'
        for key, value in type_dict.items():
            if type in value:
                type_value = key
                break
        type_values.append(type_value)

    # Combine region and current type labels
    combined_region_and_type = []
    for i, region_value in enumerate(region_values):
        combined_region_and_type.append(f'{region_value}_{type_values[i]}')

    # Create dataframe that contains the region-specific multiindex
    region_specific_index = pd.DataFrame()
    region_specific_index['segment'] = df['segment']
    region_specific_index['itype'] = combined_region_and_type
    return region_specific_index


if __name__ == '__main__':
    df_index_original = pd.read_csv('E:/cluster_seed30/preprocessed_data/membrane_currents_merged_soma/multiindex_merged_soma.csv')
    input_files_dir = 'region_specific_index/'
    df_index_region_specific = create_region_specific_index(df_index_original, input_files_dir)
