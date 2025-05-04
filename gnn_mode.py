#!/usr/bin/env python3
"""
gnn_model.py
"""
import os
import argparse
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Module, Linear, ReLU, BatchNorm1d, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import NNConv, GATConv, global_mean_pool, TopKPooling
from torch_geometric.data import DataLoader, InMemoryDataset, Data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import copy


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger("GNNModel")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger



class GraphDataset(InMemoryDataset):
    """
    In-memory dataset for graph data with device features.
    Expects pre-loaded list of Data objects and a CSV/Excel of device features.
    Applies scaling and optional augmentation.
    """
    def __init__(self,
                 graphs: list,
                 device_features_path: str,
                 transform: Optional[callable] = None,
                 augment: bool = False):
        super().__init__('.', transform)
        self.graphs = graphs
        self.device_df = pd.read_excel(device_features_path)
        self.augment = augment
        self._load()

    def _load(self):
        """Scale features and prepare dataset."""
        # Extract node, edge and target tensors
        xs = torch.cat([g.x for g in self.graphs], dim=0)
        edges = torch.cat([g.edge_attr for g in self.graphs], dim=0) if len(self.graphs[0].edge_attr) else torch.empty(0)
        ys = torch.cat([g.y for g in self.graphs], dim=0)

        # Fit scalers
        self.node_scaler = StandardScaler().fit(xs.numpy())
        if edges.nelement() > 0:
            self.edge_scaler = StandardScaler().fit(edges.numpy())
        self.target_scaler = StandardScaler().fit(ys.view(-1, 1).numpy())
        self.device_scaler = StandardScaler().fit(self.device_df.values)

        # Transform and attach device features
        scaled_device = self.device_scaler.transform(self.device_df.values)

        new_graphs = []
        for idx, g in enumerate(self.graphs):
            # Node and edge normalization
            g.x = torch.tensor(self.node_scaler.transform(g.x.numpy()), dtype=torch.float32)
            if g.edge_attr.nelement():
                g.edge_attr = torch.tensor(self.edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float32)
            # Target scaling
            g.y = torch.tensor(self.target_scaler.transform(g.y.view(-1, 1).numpy()).squeeze(), dtype=torch.float32)
            # Device features attach
            df = torch.tensor(scaled_device[idx], dtype=torch.float32)
            g.device_features = df.unsqueeze(0)

          

        self.data, self.slices = self.collate(new_graphs)

    def get(self, idx):
        data = super().get(idx)
        return data


class AttentionLayer(Module):
    """Node-level attention pooling layer."""
    def __init__(self, feature_size: int):
        super().__init__()
        self.proj = Linear(feature_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.tanh(self.proj(x)).squeeze(-1)
        weights = F.softmax(scores, dim=0).unsqueeze(-1)
        return (x * weights).sum(dim=0, keepdim=True)


class PCERegressor(Module):
    """Graph Neural Network regressor with multiple conv, attention, pooling and transformer."""
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 num_device_features: int,
                 hidden_dims: tuple = (128, 64, 32),
                 heads: int = 4):
        super().__init__()
        # Convolutional layers
        self.conv1 = NNConv(num_node_features, hidden_dims[0], self._build_nn(num_edge_features, hidden_dims[0], num_node_features), aggr='mean')
        self.conv2 = GATConv(hidden_dims[0], hidden_dims[1], heads=heads, concat=False)
        self.conv3 = NNConv(hidden_dims[1], hidden_dims[2], self._build_nn(num_edge_features, hidden_dims[2], hidden_dims[1]), aggr='mean')
        # Batch norms
        self.bn1 = BatchNorm1d(hidden_dims[0])
        self.bn2 = BatchNorm1d(hidden_dims[1])
        self.bn3 = BatchNorm1d(hidden_dims[2])
        # Pooling
        self.att_pool1 = AttentionLayer(hidden_dims[0])
        self.topk1 = TopKPooling(hidden_dims[1], ratio=0.8)
        # Transformer on global feature
        self.trans = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dims[2], nhead=heads), num_layers=2)
        # Fully connected head
        fc_in = hidden_dims[2] + num_device_features
        self.fc1 = Linear(fc_in, hidden_dims[2])
        self.fc2 = Linear(hidden_dims[2], 1)

    def _build_nn(self, in_feats, hidden, out):
        return Sequential(Linear(in_feats, hidden), ReLU(), Linear(hidden, out * hidden))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # Conv layer 1
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x1 = self.att_pool1(x)
        # Conv layer 2
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x, edge_index, edge_attr, batch, _, _ = self.topk1(x, edge_index, edge_attr, batch)
        # Conv layer 3
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        # Global mean pool
        x = global_mean_pool(x, batch)
        # Transformer encoding
        x = x.unsqueeze(1)  # [batch, seq=1, feat]
        x = self.trans(x).squeeze(1)
        # Concatenate device features
        df = data.device_features.to(x.device)
        if df.dim() == 2 and df.size(0) == x.size(0):
            x = torch.cat([x, df], dim=1)
        else:
            raise ValueError(f"Device features batch mismatch {df.size()} vs {x.size()}")
        # MLP head
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(model: Module, loader: DataLoader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch).view(-1)
        loss = criterion(preds, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model: Module, loader: DataLoader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch).view(-1)
            loss = criterion(preds, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def main(args):
    logger = setup_logger()
    # Load graphs from a serialized file or other means
    graphs = torch.load(args.graphs_path)

    dataset = GraphDataset(graphs, args.device_features, augment=args.augment)
    # Split dataset
    n_train = int(len(dataset) * args.split)
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PCERegressor(
        num_node_features=args.num_node_feats,
        num_edge_features=args.num_edge_feats,
        num_device_features=args.num_device_feats,
        hidden_dims=tuple(args.hidden_dims),
        heads=args.heads
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, test_loader, criterion, device)
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
            logger.info(f"Saved new best model at epoch {epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GNN model for PSC PCE regression.")
    parser.add_argument('--graphs_path', type=str, required=True, help='Path to serialized list of PyG Data objects')
    parser.add_argument('--device_features', type=str, required=True, help='Path to device features Excel file')
    parser.add_argument('--output', type=str, default='./outputs', help='Directory to save models/logs')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--split', type=float, default=0.7, help='Train/test split ratio')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_node_feats', type=int, default=10)
    parser.add_argument('--num_edge_feats', type=int, default=3)
    parser.add_argument('--num_device_feats', type=int, default=10)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 64, 32])
    parser.add_argument('--heads', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
