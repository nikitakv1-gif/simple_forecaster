import pandas as pd
import numpy as np 

def train_test_split(y, test_size):
    if type(test_size) == float:
        split_index = int(len(y)*test_size)
        if test_size:
            y_train, y_test = y.iloc[:-split_index, :], y.iloc[-split_index:, :]
        else:
            y_train = y
            y_test = None
    else:
        if test_size:
            y_train, y_test = y.iloc[:-test_size, :], y.iloc[-test_size:, :]
        else:
            y_train = y
            y_test = None
    return y_train, y_test
