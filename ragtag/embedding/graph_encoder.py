from typing import Any, Optional

import numpy as np

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, GINEConv, GPSConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.transforms import BaseTransform, AddLaplacianEigenvectorPE

from datasets import load_dataset

# --- Data: ZINC with Laplacian positional encodings (k eigenvectors) ---
k = 15

def dump_attrs (data):
    print("data.keys():", data.keys())
    for k in data.keys():
        v = data[k]
        if torch.is_tensor(v):
            print(f"  {k}: Tensor, shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)} value={repr(v)[:200]}")

class FixedDimLaplacianPE(BaseTransform):
    def __init__(self, k_target: int, num_nodes: int, attr_name: str = 'lap_pe', **kwargs):
        self.k_target = k_target
        self.attr_name = attr_name
        self.base = AddLaplacianEigenvectorPE(k=min(k_target, num_nodes-1), attr_name=attr_name, **kwargs)

    def forward(self, data):
        data = self.base(data)  # adds data[self.attr_name] with shape (N, <= k_target)
        pe = data[self.attr_name]
        N, d = pe.shape
        if d < self.k_target:
            pad = pe.new_zeros(N, self.k_target - d)
            pe = torch.cat([pe, pad], dim=1)
        elif d > self.k_target:
            pe = pe[:, :self.k_target]
        data[self.attr_name] = pe
        return data

def transform(data):
    return FixedDimLaplacianPE(k_target=k, num_nodes=data.num_nodes, attr_name='lap_pe', is_undirected=True, normalization='sym')(data)

train_ds = ZINC(root='data/ZINC', subset=True, split='train', pre_transform=transform)
val_ds   = ZINC(root='data/ZINC', subset=True, split='val',   pre_transform=transform)
test_ds  = ZINC(root='data/ZINC', subset=True, split='test',  pre_transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=128)
test_loader  = DataLoader(test_ds, batch_size=128)

# --- Model: Graph Transformer block + LapPE injection ---
class GT(nn.Module):
    def __init__(self, in_channels, hidden=128, heads=4, pe_dim=k, out_dim=1, num_layers=4, use_gine=False):
        super().__init__()
        self.node_in = nn.Linear(in_channels + pe_dim, hidden)
        
        conv = None
        if use_gine:
            gine_conv = GINEConv(nn=nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            ))

            conv = GPSConv(
                hidden, 
                gine_conv, 
                heads=4,
                attn_type='multihead',
                dropout=0.5
            )
        else:
            conv = TransformerConv(hidden, hidden // heads, heads=heads, beta=True, dropout=0.1)

        self.layers = nn.ModuleList([
            conv
            for _ in range(num_layers)
        ])

        # Gated global attention pooling
        self.pool_gate = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        ))
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x, edge_index, batch, lap_pe=None):
        if lap_pe is None:
            lap_pe = torch.zeros(x.size(0), 0, device=x.device)

        x = torch.cat([x, lap_pe], dim=-1)
        x = self.node_in(x)
        for conv in self.layers:
            x = x + conv(x, edge_index)
            x = F.relu(x)
        g = self.pool_gate(x, batch)
        return self.readout(g).squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
in_dim = train_ds.num_node_features
model = GT(in_channels=in_dim, pe_dim=k).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

def run_epoch(loader, train=True):
    model.train(train)
    loss_sum, n = 0, 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch, lap_pe=data.lap_pe)
        loss = F.l1_loss(pred, data.y.view_as(pred))  # ZINC is MAE
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * data.num_graphs; n += data.num_graphs
    return loss_sum / n

val_mae = 1
epoch = 0
while val_mae >= 0.1:
    epoch += 1
    train_mae = run_epoch(train_loader, True)
    val_mae   = run_epoch(val_loader, False)
    print(f"epoch {epoch:d} | train MAE {train_mae:.3f} | val MAE {val_mae:.3f}")
 