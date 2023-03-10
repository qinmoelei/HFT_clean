# we are using multi process
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm


def compress_order_book(df: pd.DataFrame):
    df = df.sort_values(by="timestamp")
    # we only need to take the snap shot of the order book at each second and that's it
    df.timestamp = (df.timestamp / 1000).astype(int)
    feature_name = df.columns
    df_refined_list = []
    for timestamp in tqdm(df.timestamp.unique()):
        single_second_information = df[df.timestamp ==
                                       timestamp].iloc[0].values
        df_refined_list.append(single_second_information)
    df_refined_list = np.array(df_refined_list)
    df = pd.DataFrame(df_refined_list, columns=feature_name)
    df.index = range(len(df))
    return df



def compress_trade_information(df):
    df_refined_list = []
    feature_name = [
        "timestamp",
        "open",
        "high",
        "close",
        "low",
        "wap",
        "buy_size",
        "buy_value",
        "buy_price",
        "sell_size",
        "sell_value",
        "sell_price",
    ]
    for timestamp in tqdm(df.timestamp.unique()):
        open = df[df.timestamp == timestamp].iloc[0].px
        close = df[df.timestamp == timestamp].iloc[-1].px
        high = np.max(df[df.timestamp == timestamp].px)
        low = np.min(df[df.timestamp == timestamp].px)
        wap = np.sum(df[df.timestamp == timestamp].px *
                     df[df.timestamp == timestamp].sz) / np.sum(
                         df[df.timestamp == timestamp].sz)
        single_information = df[df.timestamp == timestamp]
        if len(single_information[single_information.side == "buy"]) != 0:
            buy_size_sum = np.sum(
                single_information[single_information.side == "buy"]["sz"])
            buy_value_sum = np.sum(
                single_information[single_information.side == "buy"]["sz"] *
                single_information[single_information.side == "buy"]["px"])
            buy_average_price = buy_value_sum / buy_size_sum
        else:
            buy_size_sum = 0
            buy_value_sum = 0
            buy_average_price = np.nan

        if len(single_information[single_information.side == "sell"]) != 0:
            sell_size_sum = np.sum(
                single_information[single_information.side == "sell"]["sz"])
            sell_value_sum = np.sum(
                single_information[single_information.side == "sell"]["sz"] *
                single_information[single_information.side == "sell"]["px"])
            sell_average_price = sell_value_sum / sell_size_sum
        else:
            sell_size_sum = 0
            sell_value_sum = 0
            sell_average_price = np.nan

        price_information = np.array([
            timestamp, open, high, close, low, wap, buy_size_sum,
            buy_value_sum, buy_average_price, sell_size_sum, sell_value_sum,
            sell_average_price
        ])
        df_refined_list.append(price_information)
    df_refined_list = np.array(df_refined_list)
    df = pd.DataFrame(df_refined_list, columns=feature_name)
    df.index = range(len(df))

    return df


if __name__ == "__main__":
    # the original file should be stored under data/processed_data folder
    order_book = pd.read_csv("data/processed_data/order_book.csv",index_col=0)
    trade = pd.read_csv("data/processed_data/BTC-USDT.csv")
    order_book_compressed = compress_order_book(order_book)
    trade_compressed = compress_trade_information(trade)
    data_path = "data/second_data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    order_book_compressed.to_csv(os.path.join(data_path, "order_book.csv"))
    trade_compressed.to_csv(os.path.join(data_path, "BTC-USDT.csv"))
