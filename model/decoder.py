import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq_actor(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 seq_len,
                 device="cuda",
                 head=1):
        if hidden_dim % head != 0:
            raise Exception(" the dimension design is wrong")
        super(Seq_actor, self).__init__()
        self.position = torch.arange(0, seq_len).unsqueeze(0).long().to(device)
        self.register_buffer("time_position", self.position)
        self.embed_timestep = nn.Embedding(seq_len, hidden_dim).to(device)
        
        self.q_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.k_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.v_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # define the layers of the model
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, head, batch_first=True)
            for _ in range(num_layers)
        ])
        self.encoder = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ])

    def forward(self, input):
        # self.time_embeddings = self.embed_timestep(self.position)

        # the input size should be (batch,seq,input_size)
        x = input
        for layer in self.encoder:
            x = layer(x)
        # x = x + self.time_embeddings

        for q_layer, k_layer, v_layer, attention_layer in zip(
                self.q_layers, self.k_layers, self.v_layers,
                self.attention_layers):
            q = q_layer(x)
            k = k_layer(x)
            v = v_layer(x)
            x, _ = attention_layer(q, k, v)
        # pass the result through the feed-forward layers
        x = x.reshape(x.shape[0], -1)
        for layer in self.decoder:
            x = layer(x)
        # return the final output
        return x

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x