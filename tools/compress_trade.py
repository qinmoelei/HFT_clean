# we are using multi process
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import time
from tqdm import tqdm


def select_time(df):
    # we need to compress all the trade information in this 1 second and form a OHLCV information
    df.timestamp = (df.timestamp / 1000).astype(int)
    df = df[df.timestamp >= 1659283200]
    df = df[df.timestamp <= 1661961599]
    return df


def compress_trade_information(df, cpu_num, i):
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
    df = df.iloc[len(df) / cpu_num * (i - 1):len(df) / cpu_num * (i)]
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
    trades = pd.read_csv("data/processed_data/trade/BTC-USDT.csv")
    trades = select_time(trades)
    cpu_number = cpu_count()
    print("CPU内核数:{}".format(cpu_count()))
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    p = Pool(cpu_number)
    for i in range(cpu_number):
        p.apply_async(compress_trade_information, args=(i, ))
    p.close()
    p.join()
