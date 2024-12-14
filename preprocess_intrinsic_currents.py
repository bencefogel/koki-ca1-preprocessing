import numpy as np
import pandas as pd
from utils import change_unit_na, preprocess_intrinsic_currents


data_dir = 'L:/cluster_seed30/raw_data'
currents = ['nax', 'nad', 'kap', 'kad', 'kdr', 'kslow', 'car', 'passive', 'capacitive']
segment_area = pd.read_csv(data_dir + '/segment_area.csv', index_col=0)

dfs = preprocess_intrinsic_currents(data_dir, currents, segment_area)




