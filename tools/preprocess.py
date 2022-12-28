import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd


def split(data: pd.DataFrame, path: str, portion=[0.8, 0.1, 0.1]):
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

    train_data.to_csv(os.path.join(path, "train.csv"))
    valid_data.to_csv(os.path.join(path, "valid.csv"))
    test_data.to_csv(os.path.join(path, "test.csv"))
    return train_data, valid_data, test_data


def split_noramlize(data: pd.DataFrame, path: str, portion=[0.6, 0.4]):
    feature = [
        'bid1_price', 'bid1_size', 'bid2_price', 'bid2_size', 'bid3_price',
        'bid3_size', 'bid4_price', 'bid4_size', 'bid5_price', 'bid5_size',
        'ask1_price', 'ask1_size', 'ask2_price', 'ask2_size', 'ask3_price',
        'ask3_size', 'ask4_price', 'ask4_size', 'ask5_price', 'ask5_size',
        'buy_volume_oe', 'sell_volume_oe', 'bid1_size_n', 'bid2_size_n',
        'bid3_size_n', 'bid4_size_n', 'bid5_size_n', 'ask1_size_n',
        'ask2_size_n', 'ask3_size_n', 'ask4_size_n', 'ask5_size_n',
        'buy_spread_oe', 'sell_spread_oe', 'imblance_volume_oe', 'open',
        'high', 'close', 'low', 'wap', 'trade_diff', 'trade_speard', 'kmid',
        'klen', 'kmid2', 'kup', 'kup2', 'klow', 'klow2', 'ksft', 'ksft2',
        'roc_10', 'roc_30', 'roc_60', 'ma_10', 'ma_30', 'ma_60', 'std_10',
        'std_30', 'std_60', 'beta_10', 'beta_30', 'beta_60', 'max_10',
        'max_30', 'max_60', 'min_10', 'min_30', 'min_60', 'qtlu_10', 'qtlu_30',
        'qtlu_60', 'qtld_10', 'qtld_30', 'qtld_60', 'rsv_10', 'rsv_30',
        'rsv_60', 'imax_10', 'imax_30', 'imax_60', 'imin_10', 'imin_30',
        'imin_60', 'imxd_10', 'imxd_30', 'imxd_60', 'cntp_10', 'cntp_30',
        'cntp_60', 'cntn_10', 'cntn_30', 'cntn_60', 'cntd_10', 'cntd_30',
        'cntd_60'
    ]
    if not os.path.exists(path):
        os.makedirs(path)
    data = data[feature]
    train_len = int(len(data) * portion[0])
    test_len = int(len(data) * portion[1])

    train_data = data.iloc[0:train_len]
    train_data.index = range(len(train_data))
    test_data = data.iloc[train_len + 1:train_len + test_len]
    test_data.index = range(len(test_data))
    train_data["buy_price"] = train_data["ask2_price"]
    train_data["sell_price"] = train_data["bid2_price"]
    test_data["buy_price"] = test_data["ask2_price"]
    test_data["sell_price"] = test_data["ask2_price"]
    train_data.to_csv(os.path.join(path, "train.csv"))
    test_data.to_csv(os.path.join(path, "test.csv"))
    train_buy_price = train_data["ask2_price"]
    train_sell_price = train_data["bid2_price"]
    test_buy_price = test_data["ask2_price"]
    test_sell_price = test_data["bid2_price"]
    mean = train_data.mean()
    std = train_data.std()
    normalized_train = (train_data - mean) / std
    normalized_test = (test_data - mean) / std
    normalized_train["buy_price"] = train_buy_price
    normalized_train["sell_price"] = train_sell_price
    normalized_test["buy_price"] = test_buy_price
    normalized_test["sell_price"] = test_sell_price
    normalized_train.to_csv(os.path.join(path, "normalized_train.csv"))
    normalized_test.to_csv(os.path.join(path, "normalized_test.csv"))

    return train_data, test_data


def split_noramlize_preserve_size(data: pd.DataFrame,
                                  path: str,
                                  portion=[0.6, 0.4]):
    feature = [
        'bid1_price', 'bid1_size', 'bid2_price', 'bid2_size', 'bid3_price',
        'bid3_size', 'bid4_price', 'bid4_size', 'bid5_price', 'bid5_size',
        'ask1_price', 'ask1_size', 'ask2_price', 'ask2_size', 'ask3_price',
        'ask3_size', 'ask4_price', 'ask4_size', 'ask5_price', 'ask5_size',
        'buy_volume_oe', 'sell_volume_oe', 'bid1_size_n', 'bid2_size_n',
        'bid3_size_n', 'bid4_size_n', 'bid5_size_n', 'ask1_size_n',
        'ask2_size_n', 'ask3_size_n', 'ask4_size_n', 'ask5_size_n',
        'buy_spread_oe', 'sell_spread_oe', 'imblance_volume_oe', 'open',
        'high', 'close', 'low', 'wap', 'trade_diff', 'trade_speard', 'kmid',
        'klen', 'kmid2', 'kup', 'kup2', 'klow', 'klow2', 'ksft', 'ksft2',
        'roc_10', 'roc_30', 'roc_60', 'ma_10', 'ma_30', 'ma_60', 'std_10',
        'std_30', 'std_60', 'beta_10', 'beta_30', 'beta_60', 'max_10',
        'max_30', 'max_60', 'min_10', 'min_30', 'min_60', 'qtlu_10', 'qtlu_30',
        'qtlu_60', 'qtld_10', 'qtld_30', 'qtld_60', 'rsv_10', 'rsv_30',
        'rsv_60', 'imax_10', 'imax_30', 'imax_60', 'imin_10', 'imin_30',
        'imin_60', 'imxd_10', 'imxd_30', 'imxd_60', 'cntp_10', 'cntp_30',
        'cntp_60', 'cntn_10', 'cntn_30', 'cntn_60', 'cntd_10', 'cntd_30',
        'cntd_60'
    ]
    preserved_feature = [
        'bid1_price',
        'bid1_size',
        'bid2_price',
        'bid2_size',
        'bid3_price',
        'bid3_size',
        'bid4_price',
        'bid4_size',
        'bid5_price',
        'bid5_size',
        'ask1_price',
        'ask1_size',
        'ask2_price',
        'ask2_size',
        'ask3_price',
        'ask3_size',
        'ask4_price',
        'ask4_size',
        'ask5_price',
        'ask5_size',
    ]
    if not os.path.exists(path):
        os.makedirs(path)
    data = data[feature]
    train_len = int(len(data) * portion[0])
    test_len = int(len(data) * portion[1])

    train_data = data.iloc[0:train_len]
    train_data.index = range(len(train_data))
    test_data = data.iloc[train_len + 1:train_len + test_len]
    test_data.index = range(len(test_data))

    mean = train_data.mean()
    std = train_data.std()
    normalized_train = (train_data - mean) / std
    normalized_test = (test_data - mean) / std
    normalized_train[preserved_feature] = train_data[preserved_feature]
    normalized_test[preserved_feature] = test_data[preserved_feature]
    normalized_train.to_csv(os.path.join(path, "normalized_train.csv"))
    normalized_test.to_csv(os.path.join(path, "normalized_test.csv"))

    return train_data, test_data


def create_bf_orderbook(df: pd.DataFrame):

    # ================================================= Basic Features ============================================= #
    # load order_book data
    df["buy_volume_oe"] = df["bid1_size"]+df["bid2_size"] + \
        df["bid3_size"]+df["bid4_size"]+df["bid5_size"]
    df["sell_volume_oe"] = df["ask1_size"]+df["ask2_size"] + \
        df["ask3_size"]+df["ask4_size"]+df["ask5_size"]
    # normalized order distribution
    df["bid1_size_n"] = df["bid1_size"] / df["buy_volume_oe"]
    df["bid2_size_n"] = df["bid2_size"] / df["buy_volume_oe"]
    df["bid3_size_n"] = df["bid3_size"] / df["buy_volume_oe"]
    df["bid4_size_n"] = df["bid4_size"] / df["buy_volume_oe"]
    df["bid5_size_n"] = df["bid5_size"] / df["buy_volume_oe"]

    df["ask1_size_n"] = df["ask1_size"] / df["sell_volume_oe"]
    df["ask2_size_n"] = df["ask2_size"] / df["sell_volume_oe"]
    df["ask3_size_n"] = df["ask3_size"] / df["sell_volume_oe"]
    df["ask4_size_n"] = df["ask4_size"] / df["sell_volume_oe"]
    df["ask5_size_n"] = df["ask5_size"] / df["sell_volume_oe"]
    # vap prive for oe
    # df["wap1_oe"] = (df["ask1_price"] * df["bid1_size"] + df["bid1_price"] *
    #                  df["ask1_size"]) / (df["bid1_size"] + df["ask1_size"])
    # df["wap2_oe"] = (df["ask2_price"] * df["bid2_size"] + df["bid2_price"] *
    #                  df["ask2_size"]) / (df["bid2_size"] + df["ask2_size"])
    # df["wap3_oe"] = (df["ask3_price"] * df["bid3_size"] + df["bid3_price"] *
    #                  df["ask3_size"]) / (df["bid3_size"] + df["ask3_size"])
    # df["wap4_oe"] = (df["ask4_price"] * df["bid4_size"] + df["bid4_price"] *
    #                  df["ask4_size"]) / (df["bid4_size"] + df["ask4_size"])
    # df["wap5_oe"] = (df["ask5_price"] * df["bid5_size"] + df["bid5_price"] *
    #                  df["ask5_size"]) / (df["bid5_size"] + df["ask5_size"])
    # df["wap1_lgr_oe"] = np.log(df["wap1_oe"]).diff()
    # df["wap2_lgr_oe"] = np.log(df["wap2_oe"]).diff()
    # df["wap3_lgr_oe"] = np.log(df["wap3_oe"]).diff()
    # df["wap4_lgr_oe"] = np.log(df["wap4_oe"]).diff()
    # df["wap5_lgr_oe"] = np.log(df["wap5_oe"]).diff()
    # price spread
    # df["wap_balance_oe"] = np.abs(df["wap1_oe"] - df["wap5_oe"])
    df["buy_spread_oe"] = np.abs(df["bid1_price"] - df["bid5_price"])
    df["sell_spread_oe"] = np.abs(df["ask1_price"] - df["ask5_price"])
    # volume imblance
    df["imblance_volume_oe"] = (df["buy_volume_oe"] - df["sell_volume_oe"]) / (
        df["buy_volume_oe"] + df["sell_volume_oe"])
    return df


def concat_df(order_book: pd.DataFrame, trades: pd.DataFrame):
    # aligment 2 dataframe, orderbook owns the most comprehensive date and trades do not always happen between 2 timestamp
    # notice that our trades happens after the first snapshot of the order book and therefore we need to take care of it
    all_timestamp = order_book.timestamp.unique()
    order_book_feature = order_book.columns.values.tolist()
    trades_feature = trades.columns.values.tolist()
    feature_names = order_book_feature + trades_feature
    all_information = []

    for timestamp in tqdm(all_timestamp):
        single_order_book_information = order_book[order_book.timestamp ==
                                                   timestamp].values
        # the trades information contains all the information in that second and we need to use the information before the order book information
        if len(trades[trades.timestamp == timestamp - 1]) != 0:
            single_trade_information = trades[trades.timestamp == timestamp -
                                              1].values
        else:
            single_trade_information = np.full([1, len(trades_feature)],
                                               np.nan)
        single_information = np.append(single_order_book_information,
                                       single_trade_information)
        all_information.append(single_information)
    all_information = np.array(all_information)
    df = pd.DataFrame(all_information, columns=feature_names)
    df.index = range(len(df))
    return df


def clean_data(df: pd.DataFrame):
    columns = [
        'timestamp', 'symbol', 'bid1_price', 'bid1_size', 'bid2_price',
        'bid2_size', 'bid3_price', 'bid3_size', 'bid4_price', 'bid4_size',
        'bid5_price', 'bid5_size', 'ask1_price', 'ask1_size', 'ask2_price',
        'ask2_size', 'ask3_price', 'ask3_size', 'ask4_price', 'ask4_size',
        'ask5_price', 'ask5_size', 'buy_volume_oe', 'sell_volume_oe',
        'bid1_size_n', 'bid2_size_n', 'bid3_size_n', 'bid4_size_n',
        'bid5_size_n', 'ask1_size_n', 'ask2_size_n', 'ask3_size_n',
        'ask4_size_n', 'ask5_size_n', 'buy_spread_oe', 'sell_spread_oe',
        'imblance_volume_oe', 'open', 'high', 'close', 'low', 'wap',
        'buy_size', 'buy_value', 'buy_price', 'sell_size', 'sell_value',
        'sell_price'
    ]
    order_book_feature = [
        'timestamp', 'symbol', 'bid1_price', 'bid1_size', 'bid2_price',
        'bid2_size', 'bid3_price', 'bid3_size', 'bid4_price', 'bid4_size',
        'bid5_price', 'bid5_size', 'ask1_price', 'ask1_size', 'ask2_price',
        'ask2_size', 'ask3_price', 'ask3_size', 'ask4_price', 'ask4_size',
        'ask5_price', 'ask5_size', 'buy_volume_oe', 'sell_volume_oe',
        'bid1_size_n', 'bid2_size_n', 'bid3_size_n', 'bid4_size_n',
        'bid5_size_n', 'ask1_size_n', 'ask2_size_n', 'ask3_size_n',
        'ask4_size_n', 'ask5_size_n', 'buy_spread_oe', 'sell_spread_oe',
        'imblance_volume_oe'
    ]
    trader_feature = [
        'open', 'high', 'close', 'low', 'wap', 'buy_size', 'buy_value',
        'buy_price', 'sell_size', 'sell_value', 'sell_price'
    ]
    df = df.iloc[1:-1]
    df = df[columns]
    order_book_df = df[order_book_feature]
    trade_df = df[trader_feature]
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
                            columns=trader_feature)
    df_all = pd.concat([order_book_df, trade_df], axis=1)
    return df_all


    # create feature
def create_feature(df: pd.DataFrame):
    df["trade_diff"] = df["close"] - df["open"]
    df["trade_speard"] = df["high"] - df["low"]
    # intermediate values
    df['max_oc'] = df[["open", "close"]].max(axis=1)
    df['min_oc'] = df[["open", "close"]].min(axis=1)
    # features
    df['kmid'] = (df['close'] - df['open']) / df['open']
    df['klen'] = (df['high'] - df['low']) / df['open']
    df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-12)
    df['kup'] = (df['high'] - df['max_oc']) / df['open']
    df['kup2'] = (df['high'] - df['max_oc']) / (df['high'] - df['low'] + 1e-12)
    df['klow'] = (df['min_oc'] - df['low']) / df['open']
    df['klow2'] = (df['min_oc'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df['ksft'] = (2 * df['close'] - df['high'] - df['low']) / df['open']
    df['ksft2'] = (2 * df['close'] - df['high'] -
                   df['low']) / (df['high'] - df['low'] + 1e-12)
    # drop intermediate values
    df.drop(columns=['max_oc', 'min_oc'], inplace=True)
    # define rolling window list
    window = [10, 30, 60]
    # features with window
    for w in window:
        df['roc_{}'.format(w)] = df['close'].shift(w) / df['close']

    for w in window:
        df['ma_{}'.format(w)] = df['close'].rolling(w).mean() / df['close']

    for w in window:
        df['std_{}'.format(w)] = df['close'].rolling(w).std() / df['close']

    for w in window:
        df['beta_{}'.format(w)] = (df['close'].shift(w) -
                                   df['close']) / (w * df['close'])

    for w in window:
        df['max_{}'.format(w)] = df['close'].rolling(w).max() / df['close']

    for w in window:
        df['min_{}'.format(w)] = df['close'].rolling(w).min() / df['close']

    for w in window:
        df['qtlu_{}'.format(
            w)] = df['close'].rolling(w).quantile(0.8) / df['close']

    for w in window:
        df['qtld_{}'.format(
            w)] = df['close'].rolling(w).quantile(0.2) / df['close']

    # for w in window:
    #     df['rank_{}'.format(w)] = df['close'].rolling(w).rank() / w

    for w in window:
        df['rsv_{}'.format(w)] = (df['close'] - df['low'].rolling(w).min()) / (
            (df['high'].rolling(w).max() - df['low'].rolling(w).min()) + 1e-12)

    for w in window:
        df['imax_{}'.format(w)] = df['high'].rolling(w).apply(np.argmax) / w

    for w in window:
        df['imin_{}'.format(w)] = df['low'].rolling(w).apply(np.argmin) / w

    for w in window:
        df['imxd_{}'.format(w)] = (df['high'].rolling(w).apply(np.argmax) -
                                   df['low'].rolling(w).apply(np.argmin)) / w

    # for w in window:
    #     df['corr_{}'.format(w)]= df['close'].rolling(w).corr(np.log(df['volume']+1))

    # for w in window:
    #     df['cord_{}'.format(w)]= (df['close']/df['close'].shift(1)).rolling(w).corr(np.log(df['volume']/df['volume'].shift(1)+1))

    df['ret1'] = df['close'].pct_change(1)
    for w in window:
        df['cntp_{}'.format(w)] = (df['ret1'].gt(0)).rolling(w).sum() / w

    for w in window:
        df['cntn_{}'.format(w)] = (df['ret1'].lt(0)).rolling(w).sum() / w

    for w in window:
        df['cntd_{}'.format(
            w)] = df['cntp_{}'.format(w)] - df['cntn_{}'.format(w)]

    df['abs_ret1'] = np.abs(df['ret1'])
    df['pos_ret1'] = df['ret1']
    df['pos_ret1'][df['pos_ret1'].lt(0)] = 0

    df.drop(columns=['ret1', 'abs_ret1', 'pos_ret1'], inplace=True)
    df = df.iloc[max(window):]
    df.index = range(len(df))
    # df["buy_sell_spread"] = df["buy_price"] - df["sell_price"]
    return df


if __name__ == "__main__":
    order_book = pd.read_csv("data/second_data/order_book.csv", index_col=0)
    trades = pd.read_csv("data/second_data/BTC-USDT.csv", index_col=0)
    order_book = create_bf_orderbook(order_book)
    df_all = concat_df(order_book, trades)
    df_all = clean_data(df_all)
    df_all = create_feature(df_all)
    df_all.to_csv("data/second_data/df_all.csv")
    train, test = split_noramlize_preserve_size(
        df_all, path="data/settle_data_for_env1")
