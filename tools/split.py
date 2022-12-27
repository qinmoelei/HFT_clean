import numpy as np
import pandas as pd
import os


def split(data: pd.DataFrame,
          path: str,
          chunk_list=[60, 3600],
          portion=[0.8, 0.1, 0.1]):
    train_len = int(len(data) * portion[0])
    valid_len = int(len(data) * portion[1])
    test_len = int(len(data) * portion[2])
    train_data = data.iloc[0:train_len]
    train_data.index = range(len(train_data))
    valid_data = data.iloc[train_len + 1:train_len + valid_len]
    valid_data.index = range(len(valid_data))
    test_data = data.iloc[train_len + valid_len + 1:train_len + valid_len +
                          test_len]
    test_data.index = range(len(test_data))
    for chunk in chunk_list:
        path += "_{}".format(chunk)
    if not os.path.exists(path):
        os.makedirs(path)
    train_data.to_csv(os.path.join(path, "train.csv"))
    valid_data.to_csv(os.path.join(path, "valid.csv"))
    test_data.to_csv(os.path.join(path, "test.csv"))
    return train_data, valid_data, test_data



if __name__ == "__main__":
    data = pd.read_csv("data/clean_data/refined_normalized_chunk_60_3600.csv",
                       index_col=0)
    split(data, path="data/experiment")
