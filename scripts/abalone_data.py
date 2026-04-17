"""

Abalone data pre-processing

"""

import sys
sys.path.append("../projects/")

import config 
import pandas as pd

data_raw = pd.read_csv(f"{config.data_dir}/raw/abalone/abalone.data", header=None)
data_cln = data_raw.replace({'M': 1, 'F': -1, 'I': 0})
data_cln.to_csv(f"{config.data_dir}/processed/abalone/abalone.data", header=False, index=None)

X = data_cln.iloc[:, :-1].to_numpy()   # all columns except last
y = data_cln.iloc[:, -1].to_numpy()    # last column

print(X.shape,y.shape)