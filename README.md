# Preprocessing of intrinsic and synaptic current data
This repository contains code to process intrinsic and synaptic current data from raw files, merges them into a structured format with a multi-index, and saves the results as a dataframe containing the multi-index and numerical array chunks.
Executes in 25 minutes.

## **Features**

The preprocessing script saves the following variables:

1. **Segment information**: The first level of the multi-index contains segment information

2. **Current type information**: The second level of the multi-index contains current type information
   - Intrinsic currents:
      - `nax`, `nad`, `kap`, `kad`, `kdr`, `kslow`, `car`, `passive`, and `capacitive`.

   - Synaptic currents:
      - `AMPA`, `NMDA`, `GABA`, and `GABA_B`.

3. **Current values (in nA)**: Preprocessed current values are saved in chunks as arrays. 

## **Configuration and Parameters**

To preprocess the currents data, you need to configure the following parameters in the main script (`merge_dataframes.py`):

1. Set the path to the raw input data in `input_dir`:
```python
data_dir = 'L:/cluster_seed30/raw_data'
```
2. Set the path to the output directory `output_dir`:
```python
output_dir = "L:/cluster_seed30/preprocessed_data"
```

## **Performance**
- Execution time: Approximately: 25 minutes
- Output size: Approximately: 14GB

## **How to run**
1. Configure the parameters as described above.
2. Run the main preprocessing script (`merge_dataframes.py`)