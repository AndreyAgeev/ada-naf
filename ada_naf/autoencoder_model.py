import copy
import random

import numpy as np

from .base import AttentionForest
from .forests import FORESTS, ForestKind, ForestType, TaskType
from typing import Optional, Tuple, Union, Callable
from dataclasses import InitVar, dataclass, field
import logging
from time import time
from numba import njit
import torch
from .naf_nn import Autoencoder
from sklearn.utils.validation import check_random_state


@njit
def _prepare_leaf_sparse(xs, leaf_ids):
    """
    Args:
        xs: Input data of shape (n_samples, n_features).
        leaf_ids: Leaf id for each sample and tree, of shape (n_samples, n_trees)
    Returns:
        Array of shape (n_samples, n_trees, n_leaves).
    """
    # leaf_ids shape: (n_samples, n_trees)
    max_leaf_id = leaf_ids.max()
    n_leaves = max_leaf_id + 1
    n_trees = leaf_ids.shape[1]
    n_samples = xs.shape[0]
    result = np.zeros((n_samples, n_trees, n_leaves), dtype=np.uint8)
    for i in range(n_samples):
        for j in range(n_trees):
            result[i, j, leaf_ids[i, j]] = 1
    return result


@dataclass
class NAFParams:
    """Parameters of Neural Attention Forest."""
    kind: Union[ForestKind, str]
    task: TaskType
    loss: Union[str, Callable] = 'mse'
    eps: Optional[int] = None
    mode: str = 'end_to_end'
    n_epochs: int = 100
    lr: float = 1.e-3
    lam: float = 0.0
    hidden_size: int = 16
    n_layers: int = 1
    target_loss_weight: float = 1.0
    forest: dict = field(default_factory=lambda: {})
    use_weights_random_init: bool = True
    weights_init_type: str = 'default'
    random_state: Optional[int] = None
    regularization_lambda: float = 0.0

    def __post_init__(self):
        if not isinstance(self.kind, ForestKind):
            self.kind = ForestKind.from_name(self.kind)


class AutoencoderModel(AttentionForest):
    def __init__(self, params: NAFParams, run_agent=None):
        super().__init__(params)
        self.params = params
        self.forest = None
        self.run_agent = run_agent
        random.seed(params.random_state)
        np.random.seed(params.random_state)
        torch.manual_seed(params.random_state)
        self._after_init()

    def _make_nn(self, n_features):
        self._n_features = n_features
        self.nn = Autoencoder(n_features, self.params.hidden_size, self.params.random_state)

    def fit(self, x, y):
        self._make_nn(x.shape[1])

    def optimize_weights(self, X, y_orig) -> 'NeuralAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        if self.params.mode == 'end_to_end':
            self._optimize_weights_end_to_end(X, y_orig)
        elif self.params.mode == 'two_step':
            self._optimize_weights_two_step(X, y_orig)
        else:
            raise ValueError(f'Wrong mode: {self.params.mode!r}')

    def _make_loss(self):
        if callable(self.params.loss):
            return self.params.loss
        elif self.params.loss == 'mse':
            return torch.nn.MSELoss()
        raise ValueError(f'Wrong loss: {self.params.loss!r}')

    def optimize_weights_unlabeled(self, X) -> 'NeuralAttentionForest':
        # assert self.forest is not None, "Need to fit before weights optimization"
        if self.params.mode == 'end_to_end':
            self._optimize_weights_unlabeled_end_to_end(X)
        else:
            raise ValueError(f'Wrong mode: {self.params.mode!r}')

    def _optimize_weights_unlabeled_end_to_end(self, X) -> 'NeuralAttentionForest':
        from sklearn.model_selection import train_test_split
        X_train, X_val = train_test_split(X, test_size=0.2,
                                          random_state=42)  # hm this should be deleted maybe
        # X_tensor = torch.tensor(X, dtype=torch.double)

        X_tensor = torch.tensor(X_train, dtype=torch.double)
        X_tensor_val = torch.tensor(X_val, dtype=torch.double)

        optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
        loss_fn = self._make_loss()
        n_epochs = self.params.n_epochs
        best_val_loss = float('inf')

        for epoch in range(n_epochs):
            # second_y, second_xs, first_alphas, second_betas
            predictions, xs_reconstruction, _alphas, _betas = self.nn(
                X_tensor,
            )
            optim.zero_grad()
            loss = loss_fn(xs_reconstruction, X_tensor)
            loss.backward()
            optim.step()
            with torch.no_grad():
                predictions, xs_reconstruction, _alphas, _betas = self.nn(
                    X_tensor_val,
                    need_attention_weights=True,
                )
                # optim.zero_grad()
                val_loss = loss_fn(xs_reconstruction, X_tensor_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.nn.state_dict())
        self.nn = Autoencoder(self._n_features, self.params.hidden_size)
        self.nn.load_state_dict(best_state)
        return self

    def predict(self, X, need_attention_weights=False) -> np.ndarray:

        X_tensor = torch.tensor(X, dtype=torch.double)
        with torch.no_grad():
            output = self.nn(
                X_tensor,
            )
            if isinstance(output, tuple):
                output = tuple([
                    out.detach().cpu().numpy()
                    for out in output
                ])
                predictions, X_reconstruction, alphas, betas = output
            else:
                predictions = output.detach().cpu().numpy()

        if not need_attention_weights:
            return predictions
        else:
            return predictions, X_reconstruction, alphas, betas
