import os
import numpy as np
import pandas as pd


def extract_basic_feature():

    # ================================================= Basic Features ============================================= #
    # load order_book data
    if False:
        # if os.path.exists('./order_book_clean.csv'):
        order_book_clean = pd.read_csv('./order_book_clean.csv', index_col=0)
    else:
        order_book = pd.read_csv('./order_book.csv')
        order_book = order_book.sort_values(by=['timestamp']).reset_index(
            drop=True)

        time_stamp = np.floor(
            np.array(order_book['timestamp'].to_list()) / 1000)
        second_idxs = [
            time_stamp[i] > time_stamp[i - 1]
            for i in range(1, len(time_stamp), 1)
        ]
        second_idxs = [True] + second_idxs
        order_book = order_book[second_idxs].reset_index(drop=True)

        order_book_clean = order_book[[
            'timestamp', 'symbol', 'bid1_price', 'bid1_size', 'bid2_price',
            'bid2_size', 'bid3_price', 'bid3_size', 'bid4_price', 'bid4_size',
            'bid5_price', 'bid5_size', 'ask1_price', 'ask1_size', 'ask2_price',
            'ask2_size', 'ask3_price', 'ask3_size', 'ask4_price', 'ask4_size',
            'ask5_price', 'ask5_size'
        ]].reset_index(drop=True)
        order_book_clean = order_book_clean[order_book_clean['symbol'] ==
                                            'BTC-USDT'].reset_index(drop=True)
        order_book_clean.to_csv('./order_book_clean.csv')
        del order_book

    # load trade data
    if os.path.exists('./trade_clean.csv'):
        trade_clean = pd.read_csv('./trade_clean.csv', index_col=0)
    else:
        trade = pd.read_csv('./trade.csv')
        trade_clean = trade[['timestamp', 'symbol', 'px', 'sz',
                             'side']].reset_index(drop=True)
        trade_clean = trade_clean[trade_clean['symbol'] ==
                                  'BTC-USDT'].reset_index(drop=True)
        trade_clean.to_csv('./trade_clean.csv')
        del trade

    # select the range of records
    order_book_clean = order_book_clean[:2000]
    # order_book_clean = order_book_clean[(1656604800019<=order_book_clean['timestamp'])&(order_book_clean['timestamp']<=1659110399609)].reset_index(drop=True)
    print(f'the length of order_book_clean is {len(order_book_clean)}')

    # calculate transcation data
    order_book_timestamp = order_book_clean['timestamp'].to_list()
    trade_timestamp = trade_clean['timestamp'].to_list()
    print(
        f'order_book start at: {min(order_book_timestamp)}, end at {max(order_book_timestamp)}'
    )
    print(
        f'trade start at: {min(trade_timestamp)}, end at {max(trade_timestamp)}'
    )

    # append feature sell_value_sum, sell_size_sum, sell_vwap, buy_value_sum, buy_size_sum, buy_vwap
    order_book_clean['sell_value_sum'] = 0
    order_book_clean['sell_size_sum'] = 0
    order_book_clean['sell_vwap'] = 0
    order_book_clean['buy_value_sum'] = 0
    order_book_clean['buy_size_sum'] = 0
    order_book_clean['buy_vwap'] = 0

    for i in range(1, len(order_book_clean), 1):

        start_time = order_book_clean['timestamp'].iloc[i - 1]
        end_time = order_book_clean['timestamp'].iloc[i]

        # collect sell & buy records
        trade_records = trade_clean[(start_time < trade_clean['timestamp'])
                                    & (trade_clean['timestamp'] <= end_time)]

        trade_sell_records = trade_records[trade_records['side'] == 'sell']
        if len(trade_sell_records) > 0:
            sell_px = np.array(trade_sell_records['px'].to_list())
            sell_sz = np.array(trade_sell_records['sz'].to_list())

            sell_size_sum = np.sum(sell_sz)
            sell_value_sum = np.sum(sell_px * sell_sz)

            order_book_clean.loc[i, 'sell_size_sum'] = sell_size_sum
            order_book_clean.loc[i, 'sell_value_sum'] = sell_value_sum
            order_book_clean.loc[i,
                                 'sell_vwap'] = sell_value_sum / sell_size_sum

        trade_buy_records = trade_records[trade_records['side'] == 'buy']
        if len(trade_buy_records) > 0:
            buy_px = np.array(trade_buy_records['px'].to_list())
            buy_sz = np.array(trade_buy_records['sz'].to_list())

            buy_size_sum = np.sum(buy_sz)
            buy_value_sum = np.sum(buy_px * buy_sz)

            order_book_clean.loc[i, 'buy_size_sum'] = buy_size_sum
            order_book_clean.loc[i, 'buy_value_sum'] = buy_value_sum
            order_book_clean.loc[i, 'buy_vwap'] = buy_value_sum / buy_size_sum

        print(f'{i}th order_book record has been processed')

    order_book_clean.to_csv('./order_book_basic.csv')
    print('The basic feature of order_book has been extracted')
    return order_book_clean


def extract_augment_feature(order_book: pd.DataFrame):

    # ================================= basic features ================================= #
    # the augmented feature based on order_book records
    real_min = 1e-8

    buy_price_1 = np.array(order_book['bid1_price'].to_list())
    buy_price_2 = np.array(order_book['bid2_price'].to_list())
    sell_price_1 = np.array(order_book['ask1_price'].to_list())
    sell_price_2 = np.array(order_book['ask2_price'].to_list())

    buy_size_1 = np.array(order_book['bid1_size'].to_list())
    buy_size_2 = np.array(order_book['bid2_size'].to_list())
    sell_size_1 = np.array(order_book['ask1_size'].to_list())
    sell_size_2 = np.array(order_book['ask2_size'].to_list())

    wap_1 = (buy_price_1 * sell_size_1 +
             sell_price_1 * buy_size_1) / (sell_size_1 + buy_size_1 + real_min)
    wap_2 = (buy_price_2 * sell_size_2 +
             sell_price_2 * buy_size_2) / (sell_size_2 + buy_size_2 + real_min)

    log_return_wap_1 = np.log(wap_1[1:] / (wap_1[:-1] + real_min))
    log_return_wap_2 = np.log(wap_2[1:] / (wap_2[:-1] + real_min))
    log_return_buy_price_1 = np.log(buy_price_1[1:] /
                                    (buy_price_1[:-1] + real_min))
    log_return_buy_price_2 = np.log(buy_price_2[1:] /
                                    (buy_price_2[:-1] + real_min))
    log_return_sell_price_1 = np.log(sell_price_1[1:] /
                                     (sell_price_1[:-1] + real_min))
    log_return_sell_price_2 = np.log(sell_price_2[1:] /
                                     (sell_price_2[:-1] + real_min))

    order_book['wap_1'] = wap_1
    order_book['wap_2'] = wap_2

    order_book['log_return_wap_1'] = np.insert(log_return_wap_1, 0, 0)
    order_book['log_return_wap_2'] = np.insert(log_return_wap_2, 0, 0)
    order_book['log_return_buy_price_1'] = np.insert(log_return_buy_price_1, 0,
                                                     0)
    order_book['log_return_buy_price_2'] = np.insert(log_return_buy_price_2, 0,
                                                     0)
    order_book['log_return_sell_price_1'] = np.insert(log_return_sell_price_1,
                                                      0, 0)
    order_book['log_return_sell_price_2'] = np.insert(log_return_sell_price_2,
                                                      0, 0)

    # ================================= combined features ================================= #
    wap_balance = np.abs(wap_1 - wap_2)
    buy_spread = buy_price_1 - buy_price_2
    sell_spread = sell_price_1 - sell_price_2
    price_spread = 2 * (sell_price_1 - buy_price_1) / (sell_price_1 +
                                                       buy_price_1 + real_min)
    total_volumn = (buy_size_1 + buy_size_2) + (sell_size_1 + sell_size_2)
    volumn_imbalance = (sell_size_1 + sell_size_2) - (buy_size_1 + buy_size_2)

    order_book['wap_balance'] = wap_balance
    order_book['buy_spread'] = buy_spread
    order_book['sell_spread'] = sell_spread
    order_book['price_spread'] = price_spread
    order_book['total_volumn'] = total_volumn
    order_book['volumn_imbalance'] = volumn_imbalance

    # ================================= transcation features ================================= #
    sell_size_sum = np.array(order_book['sell_size_sum'].to_list())
    sell_value_sum = np.array(order_book['sell_value_sum'].to_list())
    sell_vwap = np.array(order_book['sell_vwap'].to_list())

    buy_size_sum = np.array(order_book['buy_size_sum'].to_list())
    buy_value_sum = np.array(order_book['buy_value_sum'].to_list())
    buy_vwap = np.array(order_book['buy_vwap'].to_list())

    up_limit = np.zeros_like(sell_vwap)
    down_limit = np.zeros_like(sell_vwap)
    acc_volumn = np.zeros_like(sell_vwap)
    acc_value = np.zeros_like(sell_vwap)

    period = 600  # 1 hour period

    for i in range(len(order_book)):
        sell_size_sum_chunck = sell_size_sum[max(0, i - period + 1):i + 1]
        sell_value_sum_chunck = sell_value_sum[max(0, i - period + 1):i + 1]
        sell_vwap_chunck = sell_vwap[max(0, i - period + 1):i + 1]

        buy_size_sum_chunck = buy_size_sum[max(0, i - period + 1):i + 1]
        buy_value_sum_chunck = buy_value_sum[max(0, i - period + 1):i + 1]
        buy_vwap_chunck = buy_vwap[max(0, i - period + 1):i + 1]

        up_limit[i] = max(np.max(sell_vwap_chunck), np.max(buy_vwap_chunck))
        tmp_1 = sell_vwap_chunck[np.where(sell_vwap_chunck > 0)]
        tmp_2 = buy_vwap_chunck[np.where(buy_vwap_chunck > 0)]
        if len(tmp_1) > 0 and len(tmp_2) > 0:
            down_limit[i] = min(np.min(tmp_1), np.min(tmp_2))
        elif len(tmp_1) > 0 and len(tmp_2) == 0:
            down_limit[i] = np.min(tmp_1)
        elif len(tmp_1) == 0 and len(tmp_2) > 0:
            down_limit[i] = np.min(tmp_2)
        else:
            down_limit[i] = 0
        acc_volumn[i] = np.sum(sell_size_sum_chunck) + np.sum(
            buy_size_sum_chunck)
        acc_value[i] = np.sum(sell_value_sum_chunck) + np.sum(
            buy_value_sum_chunck)

    order_book['up_limit'] = up_limit
    order_book['down_limit'] = down_limit
    order_book['acc_volumn'] = acc_volumn
    order_book['acc_value'] = acc_value

    # ================================= combined transcation features ================================= #

    order_book.to_csv('./order_book_augment.csv')
    return order_book


if __name__ == '__main__':

    # extract basic feature
    # if False:
    if os.path.exists('./order_book_basic.csv'):
        order_book_basic = pd.read_csv('./order_book_basic.csv', index_col=0)
    else:
        order_book_basic = extract_basic_feature()

    # extract augmented feature
    if False:
        # if os.path.exists('./order_book_augment.csv'):
        order_book_augment = pd.read_csv('./order_book_augmented.csv',
                                         index_col=0)
    else:
        order_book_augment = extract_augment_feature(order_book_basic)
