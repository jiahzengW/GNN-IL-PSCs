#!/usr/bin/env python3
"""
feature_analysis.py

"""
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt


def extract_embeddings(model, dataset, device, batch_size=64):
    """
    Extract graph-level embeddings from the GNN for each sample.

    Args:
        model: Trained GNN model instance
        dataset: PyG dataset of Data objects
        device: torch.device ('cpu' or 'cuda')
        batch_size: batch size for DataLoader
    Returns:
        embeddings: np.ndarray of shape (n_samples, embed_dim)
        targets: np.ndarray of shape (n_samples,)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings, targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Forward pass through GNN layers
            x = F.relu(model.bn1(model.conv1(batch.x, batch.edge_index, batch.edge_attr)))
            x = model.att1(x)
            x = F.relu(model.bn2(model.conv2(x, batch.edge_index, batch.edge_attr)))
            x = model.att2(x)
            x = F.relu(model.bn3(model.conv3(x, batch.edge_index, batch.edge_attr)))
            x = model.att3(x)
            # Pool graph
            x = global_mean_pool(x, batch.batch)
            # Transformer encoding
            x = model.transformer_layer(x).cpu().numpy()

            embeddings.append(x)
            targets.append(batch.y.cpu().numpy().reshape(-1,))

    embeddings = np.vstack(embeddings)
    targets = np.concatenate(targets)
    return embeddings, targets


def build_feature_dataframe(embeddings: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a DataFrame with embeddings and one-hot ionic liquid labels.

    Args:
        embeddings: array of shape (n_samples, embed_dim)
        metadata: DataFrame with 'Smiles_cation' and 'Smiles_anion' columns
    Returns:
        df_full: combined DataFrame for training
    """
    df_emb = pd.DataFrame(embeddings, index=metadata.index)
    df_meta = metadata[['Smiles_cation', 'Smiles_anion']]
    df_cat = pd.get_dummies(df_meta, prefix=['Cat', 'Ani'], dtype=int)
    df_full = pd.concat([df_emb, df_cat], axis=1)
    return df_full


def train_xgb(X: np.ndarray, y: np.ndarray, params: dict = None) -> xgb.XGBRegressor:
    """
    Train an XGBoost regressor on provided features.

    Args:
        X: feature matrix
        y: target vector
        params: dict of XGBoost parameters
    Returns:
        Trained XGBRegressor model
    """
    if params is None:
        params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def plot_feature_importance(model: xgb.XGBRegressor, feature_names: list, top_n: int = 20):
    """
    Plot top_n native feature importances from XGBoost.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    plt.bar(df_imp['Feature'], df_imp['Importance'])
    plt.xticks(rotation=90)
    plt.title('Top Feature Importances (XGBoost)')
    plt.tight_layout()
    plt.show()


def plot_shap_summary(model: xgb.XGBRegressor, X: pd.DataFrame):
    """
    Compute SHAP values and display a summary plot.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature analysis with GNN embeddings and XGBoost')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to Excel file with Smiles_cation, Smiles_anion, and PCE columns')
    parser.add_argument('--model_ckpt', type=str, required=True,
                        help='Path to trained GNN checkpoint (.pth)')
    parser.add_argument('--graphs', type=str, required=True,
                        help='Path to saved list of PyG Data objects')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # Load metadata and graphs
    metadata = pd.read_excel(args.metadata)
    graphs = torch.load(args.graphs)

    # Initialize model and load weights
    from gnn_model import PCERegressor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn = PCERegressor(num_node_features=metadata.shape[1], num_edge_features=metadata.shape[1], num_device_features=0).to(device)
    gnn.load_state_dict(torch.load(args.model_ckpt, map_location=device))

    # Extract embeddings & targets
    embeddings, targets = extract_embeddings(gnn, graphs, device, args.batch_size)

    # Build training DataFrame
    df_full = build_feature_dataframe(embeddings, metadata)
    X = df_full.values
    y = targets

