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
from .naf_nn import NAFNetwork
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


class NeuralAttentionForest(AttentionForest):
    def __init__(self, params: NAFParams, run_agent=None):
        self.params = params
        self.forest = None
        self.run_agent = run_agent
        random.seed(params.random_state)
        np.random.seed(params.random_state)
        self._after_init()

    def _make_nn(self, n_features):
        self._n_features = n_features
        self.nn = NAFNetwork(n_features, self.params.hidden_size, self.params.n_layers, self.params.random_state)

    def _base_fit(self, X, y) -> 'NeuralAttentionForest':
        forest_cls = FORESTS[ForestType(self.params.kind, self.params.task)]
        self.forest = forest_cls(**self.params.forest)
        self.forest.random_state = self.params.random_state
        logging.debug("Start fitting Random forest")
        start_time = time()
        self.forest.fit(X, y)
        end_time = time()
        logging.info("Random forest fit time: %f", end_time - start_time)
        # store training X and y
        self.training_xs = X.copy()
        self.training_y = self._preprocess_target(y.copy())
        # store leaf id for each point in X
        start_time = time()
        self.training_leaf_ids = self.forest.apply(self.training_xs)
        end_time = time()
        logging.info("Random forest apply time: %f", end_time - start_time)
        # make a tree-leaf-points correspondence
        logging.debug("Generating leaves data")
        start_time = time()

        self.leaf_sparse = _prepare_leaf_sparse(self.training_xs, self.training_leaf_ids)
        end_time = time()
        logging.info("Leaf generation time: %f", end_time - start_time)
        # self.tree_weights = np.ones(self.forest.n_estimators)
        logging.debug("Initializing the neural network")
        self.n_trees = self.forest.n_estimators
        self._make_nn(n_features=X.shape[1])
        return self

    def fit(self, x, y):
        self._base_fit(x, y)

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
        assert self.forest is not None, "Need to fit before weights optimization"
        if self.params.mode == 'end_to_end':
            self._optimize_weights_unlabeled_end_to_end(X)
        else:
            raise ValueError(f'Wrong mode: {self.params.mode!r}')

    def _optimize_weights_unlabeled_end_to_end(self, X) -> 'NeuralAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        from sklearn.model_selection import train_test_split
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

        neighbors_hot = self._get_leaf_data_segments(X_train, exclude_input=False)
        neighbors_hot_val = self._get_leaf_data_segments(X_val, exclude_input=False)

        X_tensor = torch.tensor(X_train, dtype=torch.double)
        X_tensor_val = torch.tensor(X_val, dtype=torch.double)

        background_X = torch.tensor(self.training_xs, dtype=torch.double)
        background_y = torch.tensor(self.training_y.data, dtype=torch.double)

        if len(background_y.shape) == 1:
            background_y = background_y.unsqueeze(1)
        neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool)
        neighbors_hot_val = torch.tensor(neighbors_hot_val, dtype=torch.bool)

        optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
        loss_fn = self._make_loss()
        n_epochs = self.params.n_epochs
        best_val_loss = float('inf')

        for epoch in range(n_epochs):
            predictions, xs_reconstruction, _alphas, _betas = self.nn(
                X_tensor,
                background_X,
                background_y,
                neighbors_hot,
                need_attention_weights=True,
            )
            optim.zero_grad()
            loss = loss_fn(xs_reconstruction, X_tensor)
            loss.backward()
            optim.step()
            # print(f"epoch = {epoch} - train {loss}")

            with torch.no_grad():
                predictions, xs_reconstruction, _alphas, _betas = self.nn(
                    X_tensor_val,
                    background_X,
                    background_y,
                    neighbors_hot_val,
                    need_attention_weights=True,
                )
                val_loss = loss_fn(xs_reconstruction, X_tensor_val)
                if val_loss < best_val_loss:
                    # print(f"epoch = {epoch} - val {val_loss}")
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.nn.state_dict())

        self.nn = NAFNetwork(self._n_features, self.params.hidden_size, self.params.n_layers)
        self.nn.load_state_dict(best_state)
        return self

    def _get_leaf_data_segments(self, X, exclude_input=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            X: Input points.
            exclude_input: Exclude leaf points that are exactly the same as input point.
                           It is useful to unbias training when fitting and optimizing
                           on the same data set.
        """
        leaf_ids = self.forest.apply(X)
        # shape of leaf_ids: (n_samples, n_trees)
        result = np.zeros((X.shape[0], self.leaf_sparse.shape[0], self.leaf_sparse.shape[1]), dtype=np.uint8)
        # shape of `self.leaf_sparse`: (n_background_samples, n_trees, n_leaves)
        for i in range(leaf_ids.shape[0]):
            for j in range(leaf_ids.shape[1]):
                result[i, :, j] = self.leaf_sparse[:, j, leaf_ids[i, j]]
            if exclude_input:
                result[i, i, :] = 0
        # result shape: (n_samples, n_background_samples, n_trees)
        return result

    def predict(self, X, need_attention_weights=False) -> np.ndarray:
        assert self.forest is not None, "Need to fit before predict"
        # all_leaf_x, all_leaf_y, sample_ids, tree_ids = self._get_leaf_data_segments(X, exclude_input=False)
        neighbors_hot = self._get_leaf_data_segments(X, exclude_input=False)
        X_tensor = torch.tensor(X, dtype=torch.double)
        background_X = torch.tensor(self.training_xs, dtype=torch.double)
        background_y = torch.tensor(self.training_y.data, dtype=torch.double)
        # size = self.training_y.shape[0]
        # background_y = torch.tensor(self.training_y.toarray(), dtype=torch.double)
        # background_y = torch.reshape(background_y, (size, 2))
        if len(background_y.shape) == 1:
            background_y = background_y.unsqueeze(1)
        neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool)
        with torch.no_grad():
            output = self.nn(
                X_tensor,
                background_X,
                background_y,
                neighbors_hot,
                need_attention_weights=need_attention_weights,
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
