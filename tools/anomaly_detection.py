import time

import numpy as np
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

from sklearn.preprocessing import MinMaxScaler

from datasets import *

from deepod.models.dif import DeepIsolationForest
from deepod.models.dsvdd import DeepSVDD
from deepod.models.icl import ICL

from ada_naf.autoencoder_model import AutoencoderModel
from ada_naf.forests import ForestKind, TaskType
from ada_naf.naf_model import NAFParams
from ada_naf.naf_model import NeuralAttentionForest
from ada_naf.naf_model_rf_multihead import NeuralMultiheadAttentionRandomForest
from logger.file_logger import FileLogger


def evaluate(y_true, scores):
    roc_auc = metrics.roc_auc_score(y_true, scores)
    ap = metrics.average_precision_score(y_true, scores)
    return roc_auc, ap


class AnomalyDetection:
    def __init__(self, num_seeds: int = 1, num_cross_val: int = 1, num_trees: int = 150, count_epoch: int = 300,
                 contaminations: int = 5):
        self._datasets = [
            DatasetArrythmia(),
            # DatasetCredit(),
            DatasetHaberman(),
            DatasetIonosphere(),
            DatasetPimaDiabetes(),
            # DatasetSeismicBumps(),
            # DatasetShuttle(),
            DatasetAnnhyroid(),
            DatasetBankAdditional(),
            DatasetCeleba()
        ]
        self._hidden_size = [8, 12, 6, 2, 2, 6, 3, 6, 6, 6]

        self._num_seeds = num_seeds
        self._num_cross_val = num_cross_val
        self._n_trees = num_trees
        self._count_epoch = count_epoch
        self._tree_type = ForestKind.RANDOM
        self._contaminations = contaminations
        self._seed_variants = [1234 + 7 * i for i in range(self._num_seeds)]
        self._file_logger = FileLogger()
        self._file_logger.setup({"num_seeds": num_seeds,
                                 "num_cross_val": num_cross_val,
                                 "num_trees": num_trees,
                                 "count_epoch": self._count_epoch,
                                 "contaminations": self._contaminations})

        self._dict_models_naf = {
            "ADA-NAF-1-LAYER": NeuralAttentionForest,
            "ADA-NAF-3-LAYER": NeuralAttentionForest,
            "ADA-NAF-MH-3-HEAD-1-LAYER": NeuralMultiheadAttentionRandomForest,
            "AUTOENCODER-1-LAYER": AutoencoderModel
        }

        self._dict_models_other = {
            "IF": (IsolationForest, {'n_estimators': num_trees, 'random_state': 42}),
            "DIF": (DeepIsolationForest, {'epochs': self._count_epoch,
                                          'n_ensemble': 6,  # тут было 50
                                          'n_estimators': num_trees,
                                          'random_state': 42}),
            "ICL": (ICL, {'epochs': self._count_epoch,
                          'device': "cpu",
                          'random_state': 42}),
            "DEEPSVD": (DeepSVDD, {'epochs': self._count_epoch, "random_state": 42, 'device': "cpu"}),
        }

    def start_model_other(self, model_type: str):
        self._file_logger.info(model_type)
        for dataset in self._datasets:
            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")
            all_F1 = []
            auc_lst, ap_lst = [], []
            t1_lst = []
            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    dataset.load()

                    model_cls = self._dict_models_other[model_type][0]
                    config_model = self._dict_models_other[model_type][1]
                    config_model["random_state"] = seed
                    model = model_cls(**config_model)

                    dataset.cross_validation_split(k)
                    start_time = time.time()
                    model.fit(dataset.X_train[dataset.y_train == 0])
                    t1 = time.time()

                    y_preds = model.predict(dataset.X_test)
                    scores = model.decision_function(dataset.X_test)
                    if model_type == "IF":
                        y_preds[y_preds == 1] = 0
                        y_preds[y_preds == -1] = 1
                        scores = -scores
                    all_F1.append(f1_score(dataset.y_test, y_preds))

                    auc, ap = evaluate(dataset.y_test, scores)
                    auc_lst.append(auc)
                    ap_lst.append(ap)
                    t1_lst.append(t1 - start_time)

            self._file_logger.info("mean test f1: " + str(np.mean(np.array(all_F1))))
            avg_auc, avg_ap = np.average(auc_lst), np.average(ap_lst)
            std_auc, std_ap = np.std(auc_lst), np.std(ap_lst)
            avg_time = np.average(t1_lst)
            txt = f'avg_auc: {avg_auc:.4f}, std_auc: {std_auc:.4f}, ' \
                  f'avg_ap: {avg_ap:.4f}, std_ap: {std_ap:.4f}, ' \
                  f'avg_time: {avg_time:.1f}'
            self._file_logger.info(txt)

    def start_model_naf(self, model_type: str, regularization_lambda: float = 0.0):
        self._file_logger.info(model_type + ": reqularization lambda = " + str(regularization_lambda))
        model_cls = self._dict_models_naf[model_type]

        for dataset, hidden_size in zip(self._datasets, self._hidden_size):
            all_F1 = []
            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")
            auc_lst, ap_lst = [], []
            t1_lst = []
            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    print(f"running: {k} and {seed}")
                    dataset.load()
                    forest_param = dict(
                        n_estimators=self._n_trees,
                        min_samples_leaf=1
                    )
                    if model_type == "ADA-NAF-3-LAYER":
                        num_layers = 3
                    else:
                        num_layers = 1
                    hidden_size = dataset.X_train.shape[1] // 2
                    params = NAFParams(
                        kind=self._tree_type,
                        task=TaskType.CLASSIFICATION,
                        mode='end_to_end',
                        n_epochs=self._count_epoch,
                        lr=0.01,
                        lam=1.0,
                        target_loss_weight=1.0,
                        hidden_size=hidden_size,
                        n_layers=num_layers,
                        forest=forest_param,
                        random_state=seed,
                        regularization_lambda=regularization_lambda
                    )
                    model = model_cls(params)

                    dataset.load()
                    dataset.cross_validation_split(k)
                    start_time = time.time()

                    model.fit(dataset.X_train, dataset.y_train)
                    model.optimize_weights_unlabeled(dataset.X_train[dataset.y_train == 0])
                    t1 = time.time()

                    dataset_test = dataset.X_test

                    y_proba, x_recons, alphas, betas = model.predict(dataset_test,
                                                                     need_attention_weights=True)
                    mse_values = np.mean(np.power(dataset_test - x_recons, 2), axis=1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scores = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))
                    y_preds = np.where(scores < 0.1, 0, 1)

                    auc, ap = evaluate(dataset.y_test, scores)
                    print(f"auc = {auc}, ap = {ap}")
                    auc_lst.append(auc)
                    ap_lst.append(ap)
                    t1_lst.append(t1 - start_time)
                    print(f1_score(dataset.y_test, y_preds))
                    all_F1.append(f1_score(dataset.y_test, y_preds))
            self._file_logger.info("mean test f1: " + str(np.mean(np.array(all_F1))))
            avg_auc, avg_ap = np.average(auc_lst), np.average(ap_lst)
            std_auc, std_ap = np.std(auc_lst), np.std(ap_lst)
            avg_time = np.average(t1_lst)
            txt = f'avg_auc: {avg_auc:.4f}, std_auc: {std_auc:.4f}, ' \
                  f'avg_ap: {avg_ap:.4f}, std_ap: {std_ap:.4f}, ' \
                  f'avg_time: {avg_time:.1f}'
            self._file_logger.info(txt)

    def start_model_naf_eps(self, model_type: str, regularization_lambda: float = 0.0):
        self._file_logger.info(model_type + ": reqularization lambda = " + str(regularization_lambda))
        model_cls = self._dict_models_naf[model_type]

        for dataset in self._datasets:
            all_F1 = []
            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")
            auc_lst, ap_lst = [[], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], []]
            t1_lst = []
            eps_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    for idx, eps in enumerate(eps_list):
                        print(f"running: {k} and {seed}")
                        dataset.load()
                        forest_param = dict(
                            n_estimators=self._n_trees,
                            min_samples_leaf=1,
                            # max_depth=1
                        )
                        if model_type == "ADA-NAF-3-LAYER":
                            num_layers = 3
                        else:
                            num_layers = 1
                        hidden_size = dataset.X_train.shape[1] // 2
                        params = NAFParams(
                            kind=self._tree_type,
                            task=TaskType.CLASSIFICATION,
                            mode='end_to_end',
                            n_epochs=self._count_epoch,
                            lr=0.01,
                            lam=1.0,
                            target_loss_weight=1.0,
                            hidden_size=hidden_size,
                            n_layers=num_layers,
                            forest=forest_param,
                            random_state=seed,
                            regularization_lambda=regularization_lambda,
                            contamination_eps=eps,
                        )
                        model = model_cls(params)

                        dataset.load()
                        dataset.cross_validation_split(k)
                        start_time = time.time()

                        model.fit(dataset.X_train, dataset.y_train)
                        model.optimize_weights_unlabeled(dataset.X_train[dataset.y_train == 0])
                        t1 = time.time()

                        dataset_test = dataset.X_test

                        y_proba, x_recons, alphas, betas = model.predict(dataset_test,
                                                                         need_attention_weights=True)
                        mse_values = np.mean(np.power(dataset_test - x_recons, 2), axis=1)
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scores = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))
                        y_preds = np.where(scores < 0.1, 0, 1)

                        auc, ap = evaluate(dataset.y_test, scores)
                        print(f"auc = {auc}, ap = {ap}")
                        auc_lst[idx].append(auc)
                        ap_lst[idx].append(ap)
                        t1_lst.append(t1 - start_time)
                        print(f1_score(dataset.y_test, y_preds))
                        all_F1.append(f1_score(dataset.y_test, y_preds))
            for idx, eps in enumerate(eps_list):
                avg_auc, avg_ap = np.average(auc_lst[idx]), np.average(ap_lst[idx])
                std_auc, std_ap = np.std(auc_lst[idx]), np.std(ap_lst[idx])
                txt = f'eps={eps} - avg_auc: {avg_auc:.4f}, std_auc: {std_auc:.4f}, ' \
                      f'avg_ap: {avg_ap:.4f}, std_ap: {std_ap:.4f}'
                self._file_logger.info(txt)

    def start_model_naf_injection(self, model_type: str, regularization_lambda: float = 0.0):
        print(model_type)
        self._file_logger.info(model_type + ": reqularization lambda = " + str(regularization_lambda))
        model_cls = self._dict_models_naf[model_type]

        for dataset, hidden_size in zip(self._datasets, self._hidden_size):
            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")
            auc_lst, ap_lst = [[], [], [], [], []], [[], [], [], [], []]
            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    # print(f"running: {k} and {seed}")
                    dataset.load()
                    forest_param = dict(
                        n_estimators=self._n_trees,
                        min_samples_leaf=1,
                        max_depth=1,
                    )
                    if model_type == "ADA-NAF-3-LAYER":
                        num_layers = 3
                    else:
                        num_layers = 1
                    hidden_size = dataset.X_train.shape[1] // 2
                    params = NAFParams(
                        kind=self._tree_type,
                        task=TaskType.CLASSIFICATION,
                        mode='end_to_end',
                        n_epochs=self._count_epoch,
                        lr=0.01,
                        lam=1.0,
                        target_loss_weight=1.0,
                        hidden_size=hidden_size,
                        n_layers=num_layers,
                        forest=forest_param,
                        random_state=seed,
                        regularization_lambda=regularization_lambda
                    )
                    dataset.cross_validation_split(k)
                    Xn, Xa = dataset.split_normal_anomalous_train()
                    datasets = dataset.create_dataset_versions(Xn, Xa, injection=True)
                    X_orig, Y_orig = dataset.create_dataset_versions(Xn, Xa, injection=True)[4]
                    for idx, subset_dataset in enumerate(datasets):
                        X, Y = subset_dataset
                        model = model_cls(params)

                        model.fit(X_orig, Y_orig)
                        model.optimize_weights_unlabeled(X)

                        dataset_test = dataset.X_test

                        y_proba, x_recons, alphas, betas = model.predict(dataset_test,
                                                                         need_attention_weights=True)
                        mse_values = np.mean(np.power(dataset_test - x_recons, 2), axis=1)
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scores = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))

                        auc, ap = evaluate(dataset.y_test, scores)
                        auc_lst[idx].append(auc)
                        ap_lst[idx].append(ap)
            for idx, subset_dataset in enumerate(datasets):
                avg_auc, avg_ap = np.average(auc_lst[idx]), np.average(ap_lst[idx])
                std_auc, std_ap = np.std(auc_lst[idx]), np.std(ap_lst[idx])
                txt = f'dataset_{idx} - avg_auc: {avg_auc:.4f}, std_auc: {std_auc:.4f}, ' \
                      f'avg_ap: {avg_ap:.4f}, std_ap: {std_ap:.4f}'
                self._file_logger.info(txt)
