import os
# Setting the LOKY_MAX_CPU_COUNT environment variable to the number of cores we need. In this case it is in 4
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


def IB2(TS, X, y):
    CS = set()

    while TS:
        x_index = TS.pop()
        x = np.array([X[x_index]])

        # Finding the nearest neighbor in CS
        NN = get_Nearest_Neighbor(x, CS, X)
        NN_target_column = y[NN]

        # If NN_target_column is not equal to x_class, we are moving x to CS
        if not np.array_equal(NN_target_column, y[x_index]):
            CS.add(x_index)

    return CS

#custom class to find the Nearest Neighbor
def get_Nearest_Neighbor(x, CS, X):
    if not CS:
        return None

    neighbor = NearestNeighbors(n_neighbors=1).fit(X[list(CS)])
    NN = neighbor.kneighbors(x, return_distance=False)[0][0]
    return list(CS)[NN]


def file_IB2(inputfile, outputfile):
    
    # Loading dataset from CSV
    df = pd.read_csv(inputfile)

    # Create a set of indices for TS
    TS = set(range(len(df)))

    # Run IB2 algorithm
    result = IB2(TS, df.iloc[:, :-1].values, df.iloc[:, -1].values)

    # Extract the reduced dataset CS
    CS_df = df.iloc[list(result)]

    # Write reduced file to disk
    CS_df.to_csv(outputfile, index=False)
    
# Running for each training dataset 
file_IB2("normalize_file_iris.csv", "IB2_iris.csv")
file_IB2("normalize_file_letter.csv", "IB2_letter.csv")