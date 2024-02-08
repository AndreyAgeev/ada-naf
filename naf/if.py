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
    forest: dict = field(default_factory=lambda:{})
    use_weights_random_init: bool = True
    weights_init_type: str = 'default'
    random_state: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.kind, ForestKind):
            self.kind = ForestKind.from_name(self.kind)


class NeuralAttentionIsolationForest(AttentionForest):
    def __init__(self, params: NAFParams, run_agent=None):
        super().__init__(params)
        self.params = params
        self.forest = None
        self.run_agent = run_agent
        self._after_init()

    def _base_fit(self, X, y) -> 'NeuralAttentionForest':
        forest_cls = IsolationForest
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
        # self.training_leaf_ids = self.forest.apply(self.training_xs)
        leaf_ids = []
        data = self.training_xs.astype('float32')
        for t in range(100):
            leaf_ids.append(self.forest.estimators_[t].tree_.apply(data))
        self.training_leaf_ids = np.array(leaf_ids)
        end_time = time()
        logging.info("Random forest apply time: %f", end_time - start_time)
        # make a tree-leaf-points correspondence
        logging.debug("Generating leaves data")
        start_time = time()
        self.training_leaf_ids = np.transpose(self.training_leaf_ids, (1, 0))
        self.leaf_sparse = _prepare_leaf_sparse(self.training_xs, self.training_leaf_ids)
        end_time = time()
        logging.info("Leaf generation time: %f", end_time - start_time)
        # self.tree_weights = np.ones(self.forest.n_estimators)
        logging.debug("Initializing the neural network")
        self.n_trees = self.forest.n_estimators
        return self

    def fit(self, x, y):
        self._base_fit(x, y)

    def optimize_weights_unlabeled(self, X) -> 'NeuralAttentionForest':
        pass

    def predict(self, X, need_attention_weights=False) -> np.ndarray:
        assert self.forest is not None, "Need to fit before predict"
        # all_leaf_x, all_leaf_y, sample_ids, tree_ids = self._get_leaf_data_segments(X, exclude_input=False)
        return self.forest.predict(X)

