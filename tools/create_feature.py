import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ! the result of the using this kind of preprocessing is a bit poor and we should use another way of doing this
# ! notice that the path for store data should be in the path of /data/sunshuo/qml
def select_feature(df: pd.DataFrame):
    # remove useless feature like the number of order in that orderbook and select a period where trades begins
    features = [
        'timestamp', 'symbol', 'bid1_price', 'bid1_size', 'bid2_price',
        'bid2_size', 'bid3_price', 'bid3_size', 'bid4_price', 'bid4_size',
        'bid5_price', 'bid5_size', 'ask1_price', 'ask1_size', 'ask2_price',
        'ask2_size', 'ask3_price', 'ask3_size', 'ask4_price', 'ask4_size',
        'ask5_price', 'ask5_size', 'buy_volume_oe', 'sell_volume_oe',
        'bid1_size_n', 'bid2_size_n', 'bid3_size_n', 'bid4_size_n',
        'bid5_size_n', 'ask1_size_n', 'ask2_size_n', 'ask3_size_n',
        'ask4_size_n', 'ask5_size_n', 'wap1_oe', 'wap2_oe', 'wap3_oe',
        'wap4_oe', 'wap5_oe', 'wap1_lgr_oe', 'wap2_lgr_oe', 'wap3_lgr_oe',
        'wap4_lgr_oe', 'wap5_lgr_oe', 'wap_balance_oe', 'buy_spread_oe',
        'sell_spread_oe', 'imblance_volume_oe', 'open', 'high', 'close', 'low',
        'wap', 'buy_size', 'buy_value', 'buy_price', 'sell_size', 'sell_value',
        'sell_price'
    ]
    df = df[features]
    df = df.iloc[1:-1]
    return df


def clean_data(df: pd.DataFrame):
    order_book_feature = [
        'timestamp', 'symbol', 'bid1_price', 'bid1_size', 'bid2_price',
        'bid2_size', 'bid3_price', 'bid3_size', 'bid4_price', 'bid4_size',
        'bid5_price', 'bid5_size', 'ask1_price', 'ask1_size', 'ask2_price',
        'ask2_size', 'ask3_price', 'ask3_size', 'ask4_price', 'ask4_size',
        'ask5_price', 'ask5_size', 'buy_volume_oe', 'sell_volume_oe',
        'bid1_size_n', 'bid2_size_n', 'bid3_size_n', 'bid4_size_n',
        'bid5_size_n', 'ask1_size_n', 'ask2_size_n', 'ask3_size_n',
        'ask4_size_n', 'ask5_size_n', 'wap1_oe', 'wap2_oe', 'wap3_oe',
        'wap4_oe', 'wap5_oe', 'wap1_lgr_oe', 'wap2_lgr_oe', 'wap3_lgr_oe',
        'wap4_lgr_oe', 'wap5_lgr_oe', 'wap_balance_oe', 'buy_spread_oe',
        'sell_spread_oe', 'imblance_volume_oe'
    ]
    trades_feature = [
        'open', 'high', 'close', 'low', 'wap', 'buy_size', 'buy_value',
        'buy_price', 'sell_size', 'sell_value', 'sell_price'
    ]
    order_book_df = df[order_book_feature]
    trade_df = df[trades_feature]
    previous_information = trade_df.iloc[0]
    previous_information["sell_price"] = previous_information["wap"]
    previous_information["buy_price"] = previous_information["wap"]
    trade_information = [previous_information.values]
    for i in range(len(trade_df) - 1):
        single_information = trade_df.iloc[i + 1]
        if single_information.isna().any():
            if single_information.isna().all():
                single_information[['open', 'high', 'close', 'low',
                                    'wap']] = previous_information["close"]
                single_information[[
                    'buy_size', 'buy_value', 'sell_size', 'sell_value'
                ]] = 0
                single_information[['buy_price', 'sell_price'
                                    ]] = previous_information["close"]
            else:
                single_information['buy_price'] = single_information['wap']
                single_information['sell_price'] = single_information['wap']
        single_trade_information = single_information.values
        trade_information.append(single_trade_information)
        previous_information = single_information
    trade_information = np.array(trade_information)
    trade_df = pd.DataFrame(trade_information,
                            index=range(1,
                                        len(order_book_df) + 1),
                            columns=trades_feature)
    df_all = pd.concat([order_book_df, trade_df], axis=1)
    return df_all


def make_chunck_feature_rolling(all_df: pd.DataFrame,
                                chunck_list: list,
                                path: str,
                                save=True):
    # make fusion feature along with the trade and order book feature
    feature_names = ['timestamp', 'symbol']
    all_df["price_spread"] = 2 * (
        all_df["ask1_price"] - all_df["bid1_price"]) / (all_df["ask1_price"] +
                                                        all_df["bid1_price"])
    all_df["volumn_imbalance"] = (
        all_df["bid1_size"] + all_df["bid2_size"] + all_df["bid3_size"] +
        all_df["bid4_size"] + all_df["bid5_size"]) - (
            all_df["ask1_size"] + all_df["ask2_size"] + all_df["ask3_size"] +
            all_df["ask4_size"] + all_df["ask5_size"])
    all_df["bid1_size_n_oe"] = all_df["bid1_size"] / (
        all_df["bid1_size"] + all_df["bid2_size"] + all_df["bid3_size"] +
        all_df["bid4_size"] + all_df["bid5_size"])
    all_df["bid2_size_n_oe"] = all_df["bid2_size"] / (
        all_df["bid1_size"] + all_df["bid2_size"] + all_df["bid3_size"] +
        all_df["bid4_size"] + all_df["bid5_size"])
    all_df["bid3_size_n_oe"] = all_df["bid3_size"] / (
        all_df["bid1_size"] + all_df["bid2_size"] + all_df["bid3_size"] +
        all_df["bid4_size"] + all_df["bid5_size"])
    all_df["bid4_size_n_oe"] = all_df["bid4_size"] / (
        all_df["bid1_size"] + all_df["bid2_size"] + all_df["bid3_size"] +
        all_df["bid4_size"] + all_df["bid5_size"])
    all_df["bid5_size_n_oe"] = all_df["bid5_size"] / (
        all_df["bid1_size"] + all_df["bid2_size"] + all_df["bid3_size"] +
        all_df["bid4_size"] + all_df["bid5_size"])
    all_df["ask1_size_n_oe"] = all_df["ask1_size"] / (
        all_df["ask1_size"] + all_df["ask2_size"] + all_df["ask3_size"] +
        all_df["ask4_size"] + all_df["ask5_size"])
    all_df["ask2_size_n_oe"] = all_df["ask2_size"] / (
        all_df["ask1_size"] + all_df["ask2_size"] + all_df["ask3_size"] +
        all_df["ask4_size"] + all_df["ask5_size"])
    all_df["ask3_size_n_oe"] = all_df["ask3_size"] / (
        all_df["ask1_size"] + all_df["ask2_size"] + all_df["ask3_size"] +
        all_df["ask4_size"] + all_df["ask5_size"])
    all_df["ask4_size_n_oe"] = all_df["ask4_size"] / (
        all_df["ask1_size"] + all_df["ask2_size"] + all_df["ask3_size"] +
        all_df["ask4_size"] + all_df["ask5_size"])
    all_df["ask5_size_n_oe"] = all_df["ask5_size"] / (
        all_df["ask1_size"] + all_df["ask2_size"] + all_df["ask3_size"] +
        all_df["ask4_size"] + all_df["ask5_size"])
    all_feature = [
        'bid1_price', 'bid1_size', 'bid2_price', 'bid2_size', 'bid3_price',
        'bid3_size', 'bid4_price', 'bid4_size', 'bid5_price', 'bid5_size',
        'ask1_price', 'ask1_size', 'ask2_price', 'ask2_size', 'ask3_price',
        'ask3_size', 'ask4_price', 'ask4_size', 'ask5_price', 'ask5_size',
        'buy_volume_oe', 'sell_volume_oe', 'bid1_size_n', 'bid2_size_n',
        'bid3_size_n', 'bid4_size_n', 'bid5_size_n', 'ask1_size_n',
        'ask2_size_n', 'ask3_size_n', 'ask4_size_n', 'ask5_size_n', 'wap1_oe',
        'wap2_oe', 'wap3_oe', 'wap4_oe', 'wap5_oe', 'wap1_lgr_oe',
        'wap2_lgr_oe', 'wap3_lgr_oe', 'wap4_lgr_oe', 'wap5_lgr_oe',
        'wap_balance_oe', 'buy_spread_oe', 'sell_spread_oe',
        'imblance_volume_oe', 'open', 'high', 'close', 'low', 'wap',
        'buy_price', 'sell_price', 'bid1_size_n_oe', 'bid2_size_n_oe',
        'bid3_size_n_oe', 'bid4_size_n_oe', 'bid5_size_n_oe', 'ask1_size_n_oe',
        'ask2_size_n_oe', 'ask3_size_n_oe', 'ask4_size_n_oe', 'ask5_size_n_oe',
        'price_spread', 'volumn_imbalance'
    ]

    trade_features_names = [
        'open', 'high', 'close', 'low', 'wap', 'buy_price', 'sell_price'
    ]

    for feature in trade_features_names:
        all_df[feature + "_diff"] = all_df[feature].diff()
        all_feature.append(feature + "_diff")

    for chunck_length in chunck_list:
        # preprocess the chunk of order book
        for feature in tqdm(all_feature):
            all_df["max_" + feature +
                   "_len{}".format(chunck_length)] = all_df[feature].rolling(
                       3, min_periods=1).max()
            all_df["min_" + feature +
                   "_len{}".format(chunck_length)] = all_df[feature].rolling(
                       3, min_periods=1).min()
            all_df["mean_" + feature +
                   "_len{}".format(chunck_length)] = all_df[feature].rolling(
                       3, min_periods=1).mean()
            all_df["std_" + feature +
                   "_len{}".format(chunck_length)] = all_df[feature].rolling(
                       3, min_periods=1).std()
            all_df["50_penctile_" + feature +
                   "_len{}".format(chunck_length)] = all_df[feature].rolling(
                       3, min_periods=1).quantile(0.5)
            all_df["25_penctile_" + feature +
                   "_len{}".format(chunck_length)] = all_df[feature].rolling(
                       3, min_periods=1).quantile(0.25)
            all_df["75_penctile_" + feature +
                   "_len{}".format(chunck_length)] = all_df[feature].rolling(
                       3, min_periods=1).quantile(0.75)
    sub_path = "chunk"
    for length in chunck_list:
        sub_path += "_{}".format(length)
    sub_path += ".csv"
    if not os.path.exists(path):
        os.makedirs(path)
    if save:
        all_df.to_csv(os.path.join(path, sub_path))
    return all_df


def select_feature_df(all_df: pd.DataFrame,
                      chunck_list: list,
                      path: str,
                      save=True):

    important_features = [
        'timestamp', 'bid1_price', 'ask1_price', 'wap', 'buy_price',
        'sell_price'
    ]
    important_df = all_df[important_features]
    basic_feature = [
        'bid1_size_n_oe', 'bid2_size_n_oe', 'bid3_size_n_oe', 'bid4_size_n_oe',
        'bid5_size_n_oe', 'ask1_size_n_oe', 'ask2_size_n_oe', 'ask3_size_n_oe',
        'ask4_size_n_oe', 'ask5_size_n_oe', 'price_spread', 'volumn_imbalance',
        'wap_balance_oe', 'buy_spread_oe', 'sell_spread_oe', 'wap1_lgr_oe',
        'wap2_lgr_oe', 'wap3_lgr_oe', 'wap4_lgr_oe', 'wap5_lgr_oe'
    ]
    basic_df = all_df[basic_feature]
    order_price_feature = ['bid1_price', 'ask1_price', 'wap1_oe']
    basic_trade_features = [
        'open', 'high', 'close', 'low', 'wap', 'buy_price', 'sell_price'
    ]

    div_feature = [feature + "_diff" for feature in basic_trade_features]
    div_df = all_df[div_feature]
    price_gap_feature = []
    price_gap_df = pd.DataFrame()
    for order_feature in order_price_feature:
        for trade_feature in basic_trade_features:
            price_gap_feature.append("gap_" + order_feature + "_" +
                                     trade_feature)
            price_gap_df[
                "gap_" + order_feature + "_" +
                trade_feature] = all_df[order_feature] - all_df[trade_feature]
    normalized_trade_feature = []
    normalized_trade_df = pd.DataFrame()

    for feature in basic_trade_features:
        for chunk_length in chunck_list:
            normalized_trade_feature.append("max_normalized_" + feature +
                                            "_{}".format(chunk_length))
            normalized_trade_feature.append("min_normalized_" + feature +
                                            "_{}".format(chunk_length))
            normalized_trade_feature.append("mean_normalized_" + feature +
                                            "_{}".format(chunk_length))
            normalized_trade_feature.append("percentile_25_normalized_" +
                                            feature +
                                            "_{}".format(chunk_length))
            normalized_trade_feature.append("percentile_50_normalized_" +
                                            feature +
                                            "_{}".format(chunk_length))
            normalized_trade_feature.append("percentile_75_normalized_" +
                                            feature +
                                            "_{}".format(chunk_length))
            normalized_trade_df["max_normalized_" + feature +
                                "_{}".format(chunk_length)] = (
                                    all_df[feature] -
                                    all_df["max_" + feature +
                                           "_len{}".format(chunk_length)])
            normalized_trade_df["min_normalized_" + feature +
                                "_{}".format(chunk_length)] = (
                                    all_df[feature] -
                                    all_df["min_" + feature +
                                           "_len{}".format(chunk_length)])
            normalized_trade_df["mean_normalized_" + feature +
                                "_{}".format(chunk_length)] = (
                                    all_df[feature] -
                                    all_df["mean_" + feature +
                                           "_len{}".format(chunk_length)])
            normalized_trade_df["percentile_25_normalized_" + feature +
                                "_{}".format(chunk_length)] = (
                                    all_df[feature] -
                                    all_df["25_penctile_" + feature +
                                           "_len{}".format(chunk_length)])
            normalized_trade_df["percentile_50_normalized_" + feature +
                                "_{}".format(chunk_length)] = (
                                    all_df[feature] -
                                    all_df["50_penctile_" + feature +
                                           "_len{}".format(chunk_length)])
            normalized_trade_df["percentile_75_normalized_" + feature +
                                "_{}".format(chunk_length)] = (
                                    all_df[feature] -
                                    all_df["75_penctile_" + feature +
                                           "_len{}".format(chunk_length)])
    df = pd.concat(
        [important_df, basic_df, div_df, normalized_trade_df, price_gap_df],
        axis=1,
        ignore_index=True)
    df.columns = important_features + basic_feature + div_feature + normalized_trade_feature + price_gap_feature
    df = df.iloc[np.max(chunck_list):]
    sub_path = "normalized_chunk"
    for length in chunck_list:
        sub_path += "_{}".format(length)
    sub_path += ".csv"
    if not os.path.exists(path):
        os.makedirs(path)
    df.index = range(len(df))
    if save:
        df.to_csv(os.path.join(path, sub_path))
    return df


def refine_df(df: pd.DataFrame, chunck_list: list, path: str, save=True):
    feature_names = [
        'timestamp', 'bid1_price', 'ask1_price', 'wap', 'buy_price',
        'sell_price', 'bid1_size_n_oe', 'bid2_size_n_oe', 'bid3_size_n_oe',
        'bid4_size_n_oe', 'bid5_size_n_oe', 'ask1_size_n_oe', 'ask2_size_n_oe',
        'ask3_size_n_oe', 'ask4_size_n_oe', 'ask5_size_n_oe', 'price_spread',
        'volumn_imbalance', 'wap_balance_oe', 'buy_spread_oe',
        'sell_spread_oe', 'wap1_lgr_oe', 'wap2_lgr_oe', 'wap3_lgr_oe',
        'wap4_lgr_oe', 'wap5_lgr_oe', 'open_diff', 'high_diff', 'close_diff',
        'low_diff', 'wap_diff', 'buy_price_diff', 'sell_price_diff',
        'max_normalized_open_60', 'min_normalized_open_60',
        'mean_normalized_open_60', 'percentile_25_normalized_open_60',
        'percentile_50_normalized_open_60', 'percentile_75_normalized_open_60',
        'max_normalized_open_3600', 'min_normalized_open_3600',
        'mean_normalized_open_3600', 'percentile_25_normalized_open_3600',
        'percentile_50_normalized_open_3600',
        'percentile_75_normalized_open_3600', 'max_normalized_high_60',
        'min_normalized_high_60', 'mean_normalized_high_60',
        'percentile_25_normalized_high_60', 'percentile_50_normalized_high_60',
        'percentile_75_normalized_high_60', 'max_normalized_high_3600',
        'min_normalized_high_3600', 'mean_normalized_high_3600',
        'percentile_25_normalized_high_3600',
        'percentile_50_normalized_high_3600',
        'percentile_75_normalized_high_3600', 'max_normalized_close_60',
        'min_normalized_close_60', 'mean_normalized_close_60',
        'percentile_25_normalized_close_60',
        'percentile_50_normalized_close_60',
        'percentile_75_normalized_close_60', 'max_normalized_close_3600',
        'min_normalized_close_3600', 'mean_normalized_close_3600',
        'percentile_25_normalized_close_3600',
        'percentile_50_normalized_close_3600',
        'percentile_75_normalized_close_3600', 'max_normalized_low_60',
        'min_normalized_low_60', 'mean_normalized_low_60',
        'percentile_25_normalized_low_60', 'percentile_50_normalized_low_60',
        'percentile_75_normalized_low_60', 'max_normalized_low_3600',
        'min_normalized_low_3600', 'mean_normalized_low_3600',
        'percentile_25_normalized_low_3600',
        'percentile_50_normalized_low_3600',
        'percentile_75_normalized_low_3600', 'max_normalized_wap_60',
        'min_normalized_wap_60', 'mean_normalized_wap_60',
        'percentile_25_normalized_wap_60', 'percentile_50_normalized_wap_60',
        'percentile_75_normalized_wap_60', 'max_normalized_wap_3600',
        'min_normalized_wap_3600', 'mean_normalized_wap_3600',
        'percentile_25_normalized_wap_3600',
        'percentile_50_normalized_wap_3600',
        'percentile_75_normalized_wap_3600', 'max_normalized_buy_price_60',
        'min_normalized_buy_price_60', 'mean_normalized_buy_price_60',
        'percentile_25_normalized_buy_price_60',
        'percentile_50_normalized_buy_price_60',
        'percentile_75_normalized_buy_price_60',
        'max_normalized_buy_price_3600', 'min_normalized_buy_price_3600',
        'mean_normalized_buy_price_3600',
        'percentile_25_normalized_buy_price_3600',
        'percentile_50_normalized_buy_price_3600',
        'percentile_75_normalized_buy_price_3600',
        'max_normalized_sell_price_60', 'min_normalized_sell_price_60',
        'mean_normalized_sell_price_60',
        'percentile_25_normalized_sell_price_60',
        'percentile_50_normalized_sell_price_60',
        'percentile_75_normalized_sell_price_60',
        'max_normalized_sell_price_3600', 'min_normalized_sell_price_3600',
        'mean_normalized_sell_price_3600',
        'percentile_25_normalized_sell_price_3600',
        'percentile_50_normalized_sell_price_3600',
        'percentile_75_normalized_sell_price_3600', 'gap_bid1_price_wap',
        'gap_ask1_price_wap'
    ]
    df = df[feature_names]
    sub_path = "refined_normalized_chunk"
    for length in chunck_list:
        sub_path += "_{}".format(length)
    sub_path += ".csv"
    if not os.path.exists(path):
        os.makedirs(path)
    df.index = range(len(df))
    if save:
        df.to_csv(os.path.join(path, sub_path))
    return df


if __name__ == "__main__":

    # df = pd.read_csv("/home/sunshuo/qml/Heta/data/second_data/all.csv",
    #                  index_col=0)
    # clean_df = select_feature(df)
    # clean_df.to_csv("data/clean_data/df.csv")
    clean_df = pd.read_csv("data/clean_data/clean.csv", index_col=0)
    chunk_df = make_chunck_feature_rolling(
        clean_df,
        chunck_list=[60, 3600],
        path="/data/sunshuo/qml/Heta/clean_data")
    chunck_df = select_feature_df(chunk_df,
                                  chunck_list=[60, 3600],
                                  path="/data/sunshuo/qml/Heta/clean_data")
    chunck_df = refine_df(chunck_df,
                          chunck_list=[60, 3600],
                          path="/data/sunshuo/qml/Heta/clean_data")

    # chunk_df.to_csv("/home/sunshuo/qml/Heta/data/clean_data/chunk.csv")
