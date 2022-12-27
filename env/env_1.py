# the difference between env_1 and env is that we consider the max size of the holding position and the overall_data could be caculated
# using the size the price
from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse
import os
# here we also release the same length as the present state
# bid 是结算方式也就是卖的价格 ask是买的价格
# 设计仓位价值结算函数 来计算仓位固定后随时间变化出现的价值变化 全用bid各档位位进行结算
# 上一时刻价值计算+变仓成本
#严格的地方 sell的时候不全挂 这样reward算少了 但是我现在不能重复性
#此版本不严格的地方 减仓后自身orderbook挂的单并不会显示到下一个时刻的orderbook上 会认为会下一时刻没啥影响 没有考虑挂单被抢的情况

parser = argparse.ArgumentParser()
parser.add_argument(
    "--df_path",
    type=str,
    default="data/settle_data_for_env1/normalized_test.csv",
    help="the path for the downloaded data to generate the environment")
parser.add_argument("--random_seed",
                    type=int,
                    default=12345,
                    help="the random seed to run the result")
parser.add_argument("--tech_indicator_list",
                    type=list,
                    default=[
                        'imblance_volume_oe',
                        'sell_spread_oe',
                        'buy_spread_oe',
                        'kmid2',
                        'bid1_size_n',
                        'ksft2',
                        'ma_10',
                        'ksft',
                        'kmid',
                        'ask1_size_n',
                        'trade_diff',
                        'qtlu_10',
                        'qtld_10',
                        'cntd_10',
                        'beta_10',
                        'roc_10',
                        'bid5_size_n',
                        'rsv_10',
                        'imxd_10',
                        'ask5_size_n',
                        'ma_30',
                        'max_10',
                        'qtlu_30',
                        'imax_10',
                        'imin_10',
                        'min_10',
                        'qtld_30',
                        'cntn_10',
                        'rsv_30',
                        'cntp_10',
                        'ma_60',
                        'max_30',
                        'qtlu_60',
                        'qtld_60',
                        'cntd_30',
                        'roc_30',
                        'beta_30',
                        'bid4_size_n',
                        'rsv_60',
                        'ask4_size_n',
                        'imxd_30',
                        'min_30',
                        'max_60',
                        'imax_30',
                        'imin_30',
                        'cntd_60',
                        'roc_60',
                        'beta_60',
                        'cntn_30',
                        'min_60',
                        'cntp_30',
                        'bid3_size_n',
                        'imxd_60',
                        'ask3_size_n',
                        'sell_volume_oe',
                        'imax_60',
                        'imin_60',
                        'cntn_60',
                        'buy_volume_oe',
                        'cntp_60',
                        'bid2_size_n',
                        'kup',
                        'bid1_size',
                        'ask1_size',
                        'std_30',
                        'ask2_size_n',
                    ],
                    help="the name of the features to predict the label")


class TradingEnv(gym.Env):
    def __init__(
        self,
        config,
        transcation_cost=0,
        back_time_length=1,
        max_holding_number=0.01,
    ):
        self.tech_indicator_list = config["tech_indicator_list"]
        self.df = pd.read_csv(config["df_path"], index_col=0)
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(back_time_length, len(self.tech_indicator_list)))
        self.terminal = False
        self.stack_length = back_time_length
        self.day = back_time_length
        self.data = self.df.iloc[self.day - back_time_length:self.day]
        self.state = self.data[self.tech_indicator_list].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        self.comission_fee = transcation_cost
        self.max_holding_number = max_holding_number

    def sell_value(self, price_information, position):
        orgional_position = position
        #use bid price and size to evaluate
        value = 0
        #position 表示剩余的单量
        for i in range(1, 6):
            if position < price_information["bid{}_size".format(i)]:
                break
            else:
                position -= price_information["bid{}_size".format(i)]
                value += price_information["bid{}_price".format(
                    i)] * price_information["bid{}_size".format(i)]
        if position > 0 and i == 5:
            print("the holding could not be sell all clear")
            #执行的单量
            actual_changed_position = orgional_position - position
        else:
            value += price_information["bid{}_price".format(i)] * position
            actual_changed_position = orgional_position

        return value, actual_changed_position

    def buy_value(self, price_information, position):
        # this value measure how much
        value = 0
        orgional_position = position
        for i in range(1, 6):
            if position < price_information["ask{}_size".format(i)]:
                break
            else:
                position -= price_information["ask{}_size".format(i)]
                value += price_information["ask{}_price".format(
                    i)] * price_information["ask{}_size".format(i)]
        if i == 5 and position > 0:
            print("the holding could not be bought all clear")
            actual_changed_position = orgional_position - position
        else:
            value += price_information["ask{}_price".format(i)] * position
            actual_changed_position = orgional_position

        return value, actual_changed_position

    def calculate_value(self, price_information, position):
        return price_information["bid1_price"] * position

    def reset(self):
        self.terminal = False
        self.stack_length = self.stack_length
        self.day = self.stack_length
        self.data = self.df.iloc[self.day - self.stack_length:self.day]
        self.state = self.data[self.tech_indicator_list].values
        self.initial_reward = 0
        self.return_rate_history = [self.initial_reward]
        self.reward_history = [0]
        self.previous_position = 0
        return self.state, {}

    def step(self, action):
        # reward is calculated using the sell_price and buy_price is only needed for
        action = action / 10
        position = self.max_holding_number * action
        self.terminal = (self.day >=
                         len(self.df.index.unique()) - 2 - self.stack_length)
        previous_position = self.previous_position
        previous_price_information = self.data.iloc[-1]
        self.day += 1
        self.data = self.df.iloc[self.day - self.stack_length:self.day]
        current_price_information = self.data.iloc[-1]
        self.state = self.data[self.tech_indicator_list].values
        self.previous_position = previous_position
        self.position = position
        if previous_position >= position:
            #hold the position or sell some position
            self.sell_size = previous_position - position

            cash, actual_position_change = self.sell_value(
                previous_price_information, self.sell_size)
            self.position = self.previous_position - actual_position_change
            previous_value = self.calculate_value(previous_price_information,
                                                  self.previous_position)
            current_value = self.calculate_value(current_price_information,
                                                 self.position)
            self.reward = current_value + cash - previous_value
            if previous_value == 0:
                return_rate = 0
            else:
                return_rate = (current_value + cash -
                               previous_value) / previous_value
            self.return_rate = return_rate
            self.reward_history.append(self.reward)
            self.return_rate_history.append(return_rate)

        if previous_position < position:
            # sell some of the position
            self.buy_size = position - previous_position
            needed_cash, actual_position_change = self.buy_value(
                previous_price_information, self.buy_size)
            self.position = self.previous_position + actual_position_change
            previous_value = self.calculate_value(previous_price_information,
                                                  self.previous_position)
            current_value = self.calculate_value(current_price_information,
                                                 self.position)
            self.reward = current_value - needed_cash - previous_value
            return_rate = (current_value - needed_cash -
                           previous_value) / (previous_value + needed_cash)

            self.return_rate_history.append(return_rate)
            self.reward_history.append(self.reward)
            self.return_rate = return_rate
            # print("buy_return_rate", return_rate)
        self.previous_position = self.position

        if np.isnan(self.return_rate):
            print(previous_value)
            print(needed_cash)
            print(previous_position)
            print(self.position)
            print(previous_price_information[[
                "ask{}_size".format(i) for i in range(5)
            ]])

            print((previous_value + needed_cash))

            raise Exception("stop")
        if self.terminal:
            return_margin = self.get_final_return_rate()
        return self.state, self.reward, self.terminal, {}

    def get_final_return_rate(self):
        return_rate_history = np.array(self.return_rate_history)
        profit_magin = np.sum(return_rate_history)
        print("the portfit margin is {}%".format(profit_magin * 100))
        return profit_magin

    def get_final_return(self):
        return_all = np.sum(self.reward_history)
        return return_all


if __name__ == "__main__":
    args = parser.parse_args()
    config = vars(args)
    seed = config["random_seed"]
    env = TradingEnv(config)
    s, _ = env.reset()
    done = False
    r_list = []
    actions = np.load("result/standard_1/{}/action.npy".format(seed))
    i = 0
    real_position = [0]
    return_rate_history = []
    while not done:
        action = actions[i]
        i = i + 1
        s, r, done, info = env.step(action)
        real_position.append(env.previous_position)
        r_list.append(r)
        print(r)
        return_rate_history.append(env.return_rate)
    path = "result/standard_env_1/{}".format(seed)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, "reward.npy"), r_list)
    np.save(os.path.join(path, "position.npy"), real_position)
    np.save(os.path.join(path, "return_rate.npy"), return_rate_history)
