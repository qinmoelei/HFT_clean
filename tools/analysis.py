import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def analysis_ic(df: pd.DataFrame, path: str, analysis_all=False):
    # this dataframe should have at least 2 columns naming buy_price and sell_price we use that to calculate the reward
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
    df["midpoint"] = (df["buy_price"] + df["sell_price"]) / 2
    return_rate = df["midpoint"].diff().tolist()
    return_rate.pop(0)
    return_rate.append(0)
    df["return"] = return_rate

    if analysis_all:
        cor_df = df.corr()
        cor_df.to_csv(path)
        return cor_df
    else:
        cor = dict()
        for f in tqdm(feature):
            correlation = df["return"].corr(df[f])
            cor["{}_r_correlation".format(f)] = correlation
        np.save(path, cor)
        return cor


if __name__ == "__main__":

    n_train = pd.read_csv("data/settle_data/normalized_train.csv", index_col=0)
    n_test = pd.read_csv("data/settle_data/normalized_test.csv", index_col=0)
    analysis_ic(n_train,
                path="data/settle_data/normalized_train.npy",
                analysis_all=False)
    analysis_ic(n_test,
                path="data/settle_data/normalized_test.npy",
                analysis_all=False)
    # pass
