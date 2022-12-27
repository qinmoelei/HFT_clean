from logging import raiseExceptions
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import sys

sys.path.append(".")
from model.encoder import *
from model.atten import *
from model.utils import *
from model.decoder import *
from model.Rein import *


# the a task we want to sign to the network is the price difference between this current timestamp and
# next important point
class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        # self.out_a = nn.Linear(hidden_nodes, N_STATES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        actions_value = self.out(x)
        # a_task = self.out_a(x)
        return actions_value


class private_net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(private_net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes + 1, N_ACTIONS)
        self.out_a = nn.Linear(hidden_nodes + 1, 1)

    def forward(self, x, previous_action):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.cat([x, previous_action], dim=1)
        actions_value = self.out(x)
        a_task = self.out_a(x)
        return actions_value, a_task


class Informer(nn.Module):
    def __init__(self,
                 enc_in,
                 c_out,
                 input_len,
                 out_len,
                 factor=5,
                 d_model=64,
                 n_heads=8,
                 e_layers=3,
                 d_layers=2,
                 d_ff=None,
                 dropout=0.0,
                 attn='prob',
                 activation='gelu',
                 output_attention=False,
                 distil=True,
                 mix=True,
                 action_number=2):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.linear1 = nn.Linear(enc_in, d_model)
        self.linear2 = nn.Linear(enc_in, d_model)
        self.actor = Seq_actor(num_layers=1,
                               input_dim=d_model,
                               hidden_dim=512,
                               output_dim=action_number,
                               seq_len=int((input_len) / (2 * (e_layers - 1))))

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder([
            EncoderLayer(AttentionLayer(Attn(
                False,
                factor,
                attention_dropout=dropout,
                output_attention=output_attention),
                                        d_model,
                                        n_heads,
                                        mix=False),
                         d_model,
                         d_ff,
                         dropout=dropout,
                         activation=activation) for l in range(e_layers)
        ], [ConvLayer(d_model)
            for l in range(e_layers - 1)] if distil else None,
                               norm_layer=torch.nn.LayerNorm(d_model))
        # Decoder for action

        # Decoder for state_embedding
        self.decoder = Decoder([
            DecoderLayer(
                AttentionLayer(Attn(True,
                                    factor,
                                    attention_dropout=dropout,
                                    output_attention=False),
                               d_model,
                               n_heads,
                               mix=mix),
                AttentionLayer(FullAttention(False,
                                             factor,
                                             attention_dropout=dropout,
                                             output_attention=False),
                               d_model,
                               n_heads,
                               mix=False),
                d_model,
                d_ff,
                dropout=dropout,
                activation=activation,
            ) for l in range(d_layers)
        ],
                               norm_layer=torch.nn.LayerNorm(d_model))
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.revin_net = RevIN(enc_in)

    def forward(self,
                x_enc,
                enc_self_mask=None,
                dec_self_mask=None,
                dec_enc_mask=None,
                rni=True):
        _, _, d = x_enc.shape
        if rni:
            x_enc = self.revin_net(x_enc, "norm")
        enc_out = self.linear1(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        action_value = self.actor(enc_out)
        dec_out = self.linear2(x_enc)
        dec_out = self.decoder(dec_out,
                               enc_out,
                               x_mask=dec_self_mask,
                               cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        if rni:
            dec_out = self.revin_net(dec_out, "denorm")
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return action_value, dec_out[:, -self.pred_len:, :], attns
        else:
            return action_value, dec_out[:, -self.pred_len:, :]  # [B, L, D]


# transformer as the state encoder instead of mlp

if __name__ == "__main__":
    state = torch.randn(32, 256, 100)
    net = Informer(enc_in=100, c_out=100, input_len=256, out_len=30)
    print(net)
    # print(net(state)[0].shape)
    # print(net(state)[1].shape)

    # print(net(state, action))
    # net = MaskedCausalAttention(100, 30, n_heads=10, drop_p=0.9)
    # output = net(state)
    # block = Block(100, 30, n_heads=10, drop_p=0.9)
    # output = block(state)
    # print(output.shape)
