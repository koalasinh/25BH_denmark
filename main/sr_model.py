# -*- coding: utf-8 -*-
# 说明：GAT + 池化 + MLP 回归模型，支持 gap/gmp/gap+gmp。
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GNNModel(nn.Module):
    def __init__(self, input_dim: int, gat_hidden: int, gat_heads: int, dropout: float,
                 pooling: str, mlp_sizes):
        super().__init__()
        self.pooling = pooling
        self.dropout = dropout

        self.gat1 = GATConv(in_channels=input_dim, out_channels=gat_hidden,
                            heads=gat_heads, dropout=dropout, concat=True)
        c1 = gat_hidden * gat_heads
        self.gat2 = GATConv(in_channels=c1, out_channels=gat_hidden,
                            heads=gat_heads, dropout=dropout, concat=True)
        c2 = gat_hidden * gat_heads

        if pooling == "gap":
            pooled_dim = c2
        elif pooling == "gmp":
            pooled_dim = c2
        elif pooling == "gap+gmp":
            pooled_dim = c2 * 2
        else:
            raise ValueError("pooling 必须是 'gap'/'gmp'/'gap+gmp'")

        layers = []
        last = pooled_dim
        for h in mlp_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "gap":
            g = global_mean_pool(x, batch)
        elif self.pooling == "gmp":
            g = global_max_pool(x, batch)
        else:
            g = torch.cat([global_mean_pool(x, batch),
                          global_max_pool(x, batch)], dim=-1)

        out = self.mlp(g).squeeze(-1)
        return out
