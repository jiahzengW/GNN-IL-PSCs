#!/usr/bin/env python3
"""
interpretability.py
"""

from typing import Optional, Tuple, List, Dict
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from captum.attr import IntegratedGradients, Saliency
import shap
import networkx as nx
import matplotlib.pyplot as plt


def check_data(data: Data) -> None:
    """Validate that Data object contains required attributes."""
    required = ['x', 'edge_index', 'edge_attr', 'y']
    for attr in required:
        if not hasattr(data, attr):
            raise AttributeError(f"Data object must have '{attr}' attribute")


class GNNInterpreter:
    """Interpreter for GNN models using gradient and perturbation methods."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def saliency_node(self, data: Data, target: Optional[int] = None) -> Tensor:
        """
        Compute node-level saliency via gradients of output w.r.t. node features.

        Args:
            data: single Data object
            target: index of output neuron for attribution
        Returns:
            Tensor of shape (num_nodes, num_features) saliency scores
        """
        check_data(data)
        x = data.x.clone().detach().requires_grad_(True)
        data.x = x
        out = self.model(data)[0]
        if target is not None:
            if target >= out.numel():
                raise IndexError("Target index exceeds output dimension")
            score = out[target]
        else:
            score = out
        score.backward(retain_graph=False)
        saliency = x.grad.abs()  # absolute gradients
        return saliency

    def integrated_gradients_node(self,
                                  data: Data,
                                  baseline: Optional[Tensor] = None,
                                  steps: int = 50,
                                  target: Optional[int] = None
                                  ) -> Tensor:
        """
        Integrated Gradients for node features.

        Args:
            data: single Data object
            baseline: tensor same shape as data.x or None for zero baseline
            steps: number of interpolation steps
            target: index of output neuron
        Returns:
            Tensor of shape (num_nodes, num_features)
        """
        check_data(data)
        ig = IntegratedGradients(self.model_forward_wrapper)
        inputs = data.x.clone().detach().requires_grad_(True)
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        attributions, _ = ig.attribute(inputs, baselines=baseline,
                                        additional_forward_args=(data, ),
                                        target=target, n_steps=steps,
                                        return_convergence_delta=True)
        return attributions.abs()

    def shap_edge_perturb(self,
                           data: Data,
                           nsamples: int = 100,
                           target: Optional[int] = None
                           ) -> List[Tuple[int, int, float]]:
        """
        Estimate edge importance via SHAP perturbation of edge attributes.

        Args:
            data: single Data object
            nsamples: number of perturbed samples
            target: output index
        Returns:
            List of tuples (edge_idx, (src, dst), shap_value)
        """
        check_data(data)
        # Flatten edge_attr to 2D
        edge_attr = data.edge_attr.cpu().numpy()
        def model_fn(edge_perturb: torch.Tensor) -> torch.Tensor:
            data.edge_attr = edge_perturb
            out = self.model(data)[0]
            if target is not None:
                return out[target].unsqueeze(0)
            return out.unsqueeze(0)
        explainer = shap.KernelExplainer(model_fn, edge_attr)
        shap_values = explainer.shap_values(edge_attr, nsamples=nsamples)
        scores = shap_values[0].mean(axis=1)  # average across features
        edge_index = data.edge_index.cpu().numpy().T
        importance = [(i, tuple(edge_index[i]), float(scores[i]))
                      for i in range(len(scores))]
        importance.sort(key=lambda x: -x[2])
        return importance

    def visualize_node_importance(self,
                                  data: Data,
                                  importance: Tensor,
                                  cmap: str = 'viridis'
                                  ) -> None:
        """
        Plot node importance on the molecular graph.

        Args:
            data: Data object
            importance: Tensor of shape (num_nodes,) or (num_nodes, num_features)
            cmap: matplotlib color map
        """
        G = to_networkx(data, to_undirected=True)
        imp = importance.sum(dim=1).cpu().numpy() if importance.dim() == 2 else importance.cpu().numpy()
        plt.figure(figsize=(6,6))
        nx.draw(G,
                with_labels=True,
                node_color=imp,
                cmap=cmap,
                node_size=300,
                edge_color='gray')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=imp.min(), vmax=imp.max()))
        plt.colorbar(sm, label='Node importance')
        plt.show()

    @staticmethod
    def model_forward_wrapper(x: Tensor, data: Data) -> Tensor:
        """
        Wrapper to allow Captum to call model with modified x.
        """
        data.x = x
        out = data.__class__ = None
        return data


# Helper to aggregate multi-level node importances

def aggregate_importance(*arrays: Tensor) -> Tensor:
    """
    Aggregate multiple importance arrays by averaging.

    Args:
        arrays: variable number of importance Tensors of shape (num_nodes,)
    Returns:
        Tensor of shape (num_nodes,)
    """
    if not arrays:
        raise ValueError("At least one importance array must be provided")
    stacked = torch.stack(arrays, dim=1)
    return torch.mean(stacked, dim=1)


# Example usage
if __name__ == '__main__':
    # Assume `model` and `data_sample` are defined
    interpreter = GNNInterpreter(model)
    sal = interpreter.saliency_node(data_sample)
    ig = interpreter.integrated_gradients_node(data_sample)
    agg = aggregate_importance(sal.sum(1), ig.sum(1))
    interpreter.visualize_node_importance(data_sample, agg)
