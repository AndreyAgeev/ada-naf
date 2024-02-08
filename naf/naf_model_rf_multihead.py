import copy
import random

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import AttentionForest
from .forests import FORESTS, ForestKind, ForestType, TaskType
from typing import Optional, Tuple, Union, Callable
from dataclasses import InitVar, dataclass, field
import logging
from time import time
from numba import njit
import torch
from .naf_nn import NAFMultiheadNetwork
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


class NeuralMultiheadAttentionRandomForest(AttentionForest):
    def __init__(self, params: NAFParams, run_agent=None):
        super().__init__(params)
        self.params = params
        self.forest = None
        self.run_agent = run_agent
        random.seed(params.random_state)
        np.random.seed(params.random_state)
        self._after_init()

    def _make_nn(self, n_features):
        self._n_features = n_features
        self.nn = NAFMultiheadNetwork(n_features, self.params.hidden_size, self.params.n_layers, 3, self.params.random_state)
        if self.params.use_weights_random_init:
            MAX_INT = np.iinfo(np.int32).max
            rng = check_random_state(self.params.random_state)
            seed = rng.randint(MAX_INT)
            torch.manual_seed(seed)

            def _init_weights(m):
                if isinstance(m, torch.nn.Linear):
                    # torch.nn.init.uniform_(m.weight)
                    if self.params.weights_init_type == 'xavier':
                        torch.nn.init.xavier_normal_(m.weight)
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == 'uniform':
                        torch.nn.init.uniform_(m.weight)
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == 'general_rule_uniform':
                        n = m.in_features
                        y = 1.0 / np.sqrt(n)
                        m.weight.data.uniform_(-y, y)
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == 'general_rule_normal':
                        y = m.in_features
                        m.weight.data.normal_(0.0, 1.0 / np.sqrt(y))
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == 'default':
                        m.reset_parameters()
                    else:
                        raise ValueError(f'Wrong {self.params.weights_init_type=}')

            # self.nn.apply(_init_weights)

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
        X_train, X_val = train_test_split(X, test_size=0.2,
                                          random_state=42)  # hm this should be deleted maybe

        neighbors_hot = self._get_leaf_data_segments(X_train, exclude_input=False)
        neighbors_hot_val = self._get_leaf_data_segments(X_val, exclude_input=False)

        X_tensor = torch.tensor(X_train, dtype=torch.double)
        X_tensor_val = torch.tensor(X_val, dtype=torch.double)
        background_X = torch.tensor(self.training_xs, dtype=torch.double)
        background_y = torch.tensor(self.training_y, dtype=torch.double)

        if len(background_y.shape) == 1:
            background_y = background_y.unsqueeze(1)
        neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool)
        neighbors_hot_val = torch.tensor(neighbors_hot_val, dtype=torch.bool)

        optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
        loss_fn = self._make_loss()
        n_epochs = self.params.n_epochs
        best_val_loss = float('inf')

        for epoch in range(n_epochs):
            # second_y, second_xs, first_alphas, second_betas
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
            with torch.no_grad():
                predictions, xs_reconstruction, _alphas, _betas = self.nn(
                    X_tensor_val,
                    background_X,
                    background_y,
                    neighbors_hot_val,
                    need_attention_weights=True,
                )
                # optim.zero_grad()
                val_loss = loss_fn(xs_reconstruction, X_tensor_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.nn.state_dict())
        self.nn = NAFMultiheadNetwork(self._n_features, self.params.hidden_size, self.params.n_layers, 3)
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
        # leaf_ids = self.forest.apply(X)
        leaf_ids = []
        data = X.astype('float32')
        for t in range(100):
            leaf_ids.append(self.forest.estimators_[t].tree_.apply(data))
        leaf_ids = np.array(leaf_ids)
        leaf_ids = np.transpose(leaf_ids, (1, 0))
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
        background_y = torch.tensor(self.training_y, dtype=torch.double)
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
