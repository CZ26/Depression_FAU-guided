import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GAT import GAT
from .Predictor import Predictor
from .functions import batch_graphify
import mdg

log = mdg.utils.get_logger()

import math


class MulMoAttn(nn.Module):
    def __init__(self, in_channels, args):
        super(MulMoAttn, self).__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** args.scale_para
        self.attend = nn.Softmax(dim=-1)

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, assis, main):
        query = self.linear_q(assis)
        key = self.linear_k(main)
        value = self.linear_v(main)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, value)
        return attn, out


class MDG(nn.Module):

    def __init__(self, args):
        super(MDG, self).__init__()
        self.args = args
        self.n_attn = 1
        u_dim_a, u_dim_cnnres, u_dim_auspec = 100, 2048, 48
        g_dim = 300
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100
        tag_size = 1
        n_head = 4

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn_a = SeqContext(u_dim_a, g_dim, args)
        self.rnn_cnnres = SeqContext(u_dim_cnnres, g_dim, args)
        self.rnn_auspec = SeqContext(u_dim_auspec, g_dim, args)
        
        self.mmattn = MulMoAttn(g_dim, args)
        self.edge_att = EdgeAtt(g_dim*3, args)
        self.gat = GAT(g_dim*3, h1_dim, h2_dim, n_head)
        self.pred = Predictor(g_dim*3 + h2_dim, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        node_features_2DCNN = self.rnn_cnnres(data["len_tensor"], data["cnnresnet_tensor"]) # [batch_size, mx_len, D_g]
        node_features_A = self.rnn_a(data["len_tensor"], data["audio_tensor"]) # [batch_size, mx_len, D_g]
        node_features_au = self.rnn_auspec(data["len_tensor"], data["auspecspec_tensor"]) # [batch_size, mx_len, D_g]

        for _ in range(self.n_attn):
            score1, node_features_2DCNN = self.mmattn(node_features_2DCNN, node_features_au)
            score2, node_features_A = self.mmattn(node_features_A, node_features_au)
            node_features_2DCNN = node_features_2DCNN + node_features_au
            node_features_A = node_features_2DCNN + node_features_au

        node_features = torch.cat((node_features_2DCNN, node_features_A, node_features_au), 2)

        features, edge_index, edge_norm, edge_type, edge_index_lengths, edge_weights = batch_graphify(
            node_features, data["len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)

        graph_out = self.gat(features, edge_index)

        return graph_out, features, edge_norm, edge_weights

    def forward(self, data):
        graph_out, features, edge_norm, edge_weights = self.get_rep(data)
        out = self.pred(torch.cat([features, graph_out], dim=-1), data["len_tensor"])
        
        return out

    def get_loss(self, data):
        graph_out, features, edge_norm, edge_weights = self.get_rep(data)
        loss = self.pred.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["len_tensor"])

        return loss, edge_norm, edge_weights
