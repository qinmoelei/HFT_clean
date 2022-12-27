import numpy as np
import pandas as pd
import torch


def create_demonstration(df: pd.DataFrame, date_length):
    # the gym env is to using and df and here is to create a action list that maximize the
    # reward
    # notice that this demonstration action is designed for 0 transcation-commision fee
    action_list = [0]
    df = df.iloc[date_length - 1:]
    for i in range(len(df) - 1):
        if df.iloc[i]["ask1_price"] < df.iloc[i + 1]["ask1_price"]:
            action = 1
        else:
            if df.iloc[i]["ask1_price"] == df.iloc[i + 1]["ask1_price"]:
                action = action_list[-1]
            else:
                action = 0
        action_list.append(action)
    return action_list[1:]
