#!/usr/bin/env python3
"""
train_and_tune.py
"""
import os
import argparse
import logging
from itertools import product

import torch
from torch.optim import Adam
import torch.nn as nn
from torch_geometric.data import DataLoader
import pandas as pd
from sklearn.model_selection import ParameterGrid

# Import your custom model and dataset
from gnn_model import PCERegressor, GraphDataset


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger for console output."""
    logger = logging.getLogger("TrainAndTune")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def train(model: nn.Module,
          loader: DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device) -> float:
    """
    Single epoch training.
    Returns average loss over the loader.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch).view(-1)
        loss = criterion(preds, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> float:
    """
    Single epoch evaluation.
    Returns average loss over the loader.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch).view(-1)
            loss = criterion(preds, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and tune GNN PCE regressor.")
    parser.add_argument('--graphs', type=str, required=True,
                        help='Path to serialized list of Data objects (torch.save)')
    parser.add_argument('--device_features', type=str, required=True,
                        help='Path to device features Excel file')
    parser.add_argument('--output', type=str, default='hyperparameter_results.xlsx',
                        help='Excel file to save grid search results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--split', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs per config')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    # Load raw graph list
    graphs = torch.load(args.graphs)
    logger.info(f"Loaded {len(graphs)} graphs")

    # Prepare dataset with scaling and optional augmentation
    dataset = GraphDataset(graphs,
                           device_features_path=args.device_features,
                           augment=args.augment)
    n_train = int(len(dataset) * args.split)
    train_ds = dataset[:n_train]
    test_ds = dataset[n_train:]
    logger.info(f"Train split: {len(train_ds)}, Test split: {len(test_ds)}")

    # Define hyperparameter grid
    param_grid = {
        'lr': [0.001, 0.003, 0.01, 0.04],
        'dropout': [0.0, 0.1, 0.2, 0.3],
        'hidden_dim': [10, 48, 96, 112, 128],
        'attention_dim': [32, 64, 96]
    }

    grid = list(ParameterGrid(param_grid))
    logger.info(f"Total configurations to test: {len(grid)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    results = []
    criterion = nn.MSELoss()

    for config in grid:
        logger.info(f"Testing config: {config}")
        # DataLoader for each run (reshuffle only train)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # Initialize model
        model = PCERegressor(
            num_node_features=graphs[0].x.size(1),
            num_edge_features=graphs[0].edge_attr.size(1),
            num_device_features=dataset.device_df.shape[1],
            hidden_dims=(config['hidden_dim'], config['hidden_dim'], config['hidden_dim']),
            heads=4
        ).to(device)

        # Override dropout for each conv layer if applicable
        for m in model.modules():
            if hasattr(m, 'dropout'):  # e.g., nn.Dropout layers
                m.dropout = config['dropout']

        optimizer = Adam(model.parameters(), lr=config['lr'])

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, test_loader, criterion, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            logger.debug(f"Epoch {epoch}/{args.epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        accuracy = 1.0 - best_val_loss
        results.append({**config, 'best_val_loss': best_val_loss, 'accuracy': accuracy})
        logger.info(f"Result - Loss: {best_val_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel(args.output, index=False)
    logger.info(f"Grid search complete. Results saved to {args.output}")


if __name__ == '__main__':
    main()
