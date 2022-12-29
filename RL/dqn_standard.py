import sys

sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter
from RL.utili import Multi_step_ReplayBuffer
import random
from tqdm import tqdm
import argparse
from model.net import *
import numpy as np
import torch
from torch import nn
import yaml
import os
import pandas as pd
from env.env_1 import TradingEnv

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",
                    type=int,
                    default=4096,
                    help="the number of transcation we store in one memory")

# the replay buffer get cleared get every time the target net get updated
parser.add_argument("--batch_size",
                    type=int,
                    default=2048,
                    help="the number of transcation we learn at a time")

parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate")

parser.add_argument("--epsilon",
                    type=float,
                    default=0.95,
                    help="the learning rate")

parser.add_argument("--gamma",
                    type=float,
                    default=0.99,
                    help="the learning rate")

parser.add_argument(
    "--target_freq",
    type=int,
    default=100,
    help=
    "the number of updates before the eval could be as same as the target and clear all the replay buffer"
)
parser.add_argument(
    "--num_epoch",
    type=int,
    default=10,
    help="the number of epoch we use to pretain and train the agent")
parser.add_argument(
    "--transcation_cost",
    type=float,
    default=0.0002,
    help="the transcation cost of not holding the same action as before")
# view this as a a task

parser.add_argument("--back_time_length",
                    type=int,
                    default=1,
                    help="the length of the holding period")
"""notice that since it is a informer sctructure problem which involes twice conv on the time series to compress,
therefore back_time_length must be larger than or equal 4"""
parser.add_argument("--train_env_config",
                    type=str,
                    default="config/settle/train.yml",
                    help="the dict for storing env config")
parser.add_argument("--test_env_config",
                    type=str,
                    default="config/settle_1/test.yml",
                    help="the dict for storing env config")

parser.add_argument(
    "--result_path",
    type=str,
    default="result/standard_env_1/",
    help="the path for storing the test result",
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="the random seed for training and sample",
)

parser.add_argument(
    "--n_step",
    type=int,
    default=1,
    help="the number of step we have in the td error and replay buffer",
)


def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.seed = args.seed
        seed_torch(self.seed)
        self.model_path = os.path.join(args.result_path, str(self.seed))
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.train_ev_instance = TradingEnv(
            read_yaml_to_dict(args.train_env_config),
            transcation_cost=args.transcation_cost,
            back_time_length=args.back_time_length,
        )
        self.test_ev_instance = TradingEnv(
            read_yaml_to_dict(args.test_env_config),
            transcation_cost=args.transcation_cost,
            back_time_length=args.back_time_length,
        )
        # here is where we define the difference among different net
        self.n_action = self.train_ev_instance.action_space.n
        self.n_state = self.train_ev_instance.reset()[0].reshape(-1).shape[0]
        self.eval_net, self.target_net = Net(
            self.n_state, self.n_action, 128).to(self.device), Net(
                self.n_state, self.n_action,
                128).to(self.device)  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(),
            lr=args.lr)  # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.epsilon = args.epsilon
        self.target_freq = args.target_freq
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.result_path = os.path.join(self.model_path, "test_result")
        self.model_path = os.path.join(self.model_path, "trained_model")
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.n_step = args.n_step
        self.num_epoch = args.num_epoch
        self.buffer_size = args.buffer_size
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.target_freq, gamma=0.999)

    def update(self, states: torch.tensor, actions: torch.tensor,
               rewards: torch.tensor, next_states: torch.tensor,
               dones: torch.tensor):

        b, s, d_in = states.shape
        q_eval = self.eval_net(states.reshape(b, -1)).gather(1, actions)

        q_next = self.target_net(next_states.reshape(b, -1)).detach()


        q_target = rewards + \
            torch.max(q_next, 1)[0].view(self.batch_size, 1)*(1-dones)
        td_error = self.loss_func(q_eval, q_target)
        # print(td_error)
        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()
        self.update_counter += 1
        if self.update_counter % self.target_freq == 1:
            self.hardupdate()
            self.scheduler.step()
        return td_error.cpu()

    def hardupdate(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def act(self, state):

        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1),
                            0).to(self.device)
        if np.random.uniform() < self.epsilon:
            actions_value = self.target_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_action)
        return action

    def act_test(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1),
                            0).to(self.device)
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def train_with_valid(self):

        valid_score_list = []
        for i in range(self.num_epoch):
            replay_buffer = Multi_step_ReplayBuffer(
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                device=self.device,
                seed=self.seed,
                gamma=self.gamma,
                n_step=self.n_step,
            )
            print('<<<<<<<<<Episode: %s' % i)
            td_error_list = []
            s, _ = self.train_ev_instance.reset()
            episode_reward_sum = 0
            step_counter = 0
            while True:
                a = self.act(s)
                s_, r, done, info = self.train_ev_instance.step(a)
                replay_buffer.add(s, a, r, s_, done)
                episode_reward_sum += r
                s = s_
                step_counter += 1
                if step_counter % self.buffer_size == int(0.5 *
                                                          self.buffer_size):
                    states, actions, rewards, next_states, dones = replay_buffer.sample(
                    )
                    td_error = self.update(states, actions, rewards,
                                           next_states, dones)
                    self.writer.add_scalar(tag="TD error",
                                           scalar_value=td_error,
                                           global_step=self.update_counter,
                                           walltime=None)
                    td_error_list.append(td_error.detach().numpy())
                if done:
                    print('episode%s---reward_sum: %s' %
                          (i, round(episode_reward_sum, 2)))
                    break
            return_sum = self.train_ev_instance.get_final_return_rate()
            self.writer.add_scalar(tag="return_rate_train",
                                   scalar_value=return_sum,
                                   global_step=i,
                                   walltime=None)
            self.writer.add_scalar(tag="reward_sum_train",
                                   scalar_value=episode_reward_sum,
                                   global_step=i,
                                   walltime=None)
            td_error_mean = np.mean(td_error_list)
            self.writer.add_scalar(tag="TD error epoch",
                                   scalar_value=td_error_mean,
                                   global_step=i,
                                   walltime=None)

            all_model_path = self.model_path + "/all_model/"
            if not os.path.exists(all_model_path):
                os.makedirs(all_model_path)
            torch.save(self.eval_net.state_dict(),
                       all_model_path + "num_epoch_{}.pkl".format(i))
            s, _ = self.test_ev_instance.reset()
            episode_reward_sum = 0
            done = False
            while not done:
                a = self.act_test(s)
                s_, r, done, _ = self.test_ev_instance.step(a)
                episode_reward_sum += r
                s = s_
            return_rate = self.test_ev_instance.get_final_return_rate()
            valid_score_list.append(episode_reward_sum)
            self.writer.add_scalar(tag="return_rate_test",
                                   scalar_value=return_rate,
                                   global_step=i,
                                   walltime=None)
            self.writer.add_scalar(tag="reward_sum_test",
                                   scalar_value=episode_reward_sum,
                                   global_step=i,
                                   walltime=None)
        index = valid_score_list.index(np.max(valid_score_list))
        model_path = all_model_path + "num_epoch_{}.pkl".format(index)
        self.eval_net.load_state_dict(torch.load(model_path))
        self.hardupdate()
        best_model_path = self.model_path + "/best_model/"
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        torch.save(self.eval_net.state_dict(),
                   best_model_path + "best_model.pkl")

    def test(self):
        # self.eval_net.load_state_dict(
        #     torch.load(self.model_path + "/best_model/" + "best_model.pkl"))
        s, _ = self.test_ev_instance.reset()
        done = False
        action_list = []
        reward_list = []
        while not done:
            a = self.act_test(s)
            s_, r, done, _ = self.test_ev_instance.step(a)
            reward_list.append(r)
            s = s_
            action_list.append(a)
        return_sum = self.test_ev_instance.get_final_return_rate()
        print(return_sum)
        self.result_path = "result/standard_1/{}".format(self.seed)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # self.writer.add_scalar(tag="return_rate_sum",
        #                        scalar_value=return_sum,
        #                        global_step=i,
        #                        walltime=None)

        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        np.save(os.path.join(self.result_path, "action.npy"), action_list)
        np.save(os.path.join(self.result_path, "reward.npy"), reward_list)


if __name__ == "__main__":
    args = parser.parse_args()
    agent = DQN(args)
    agent.train_with_valid()
    agent.test()
