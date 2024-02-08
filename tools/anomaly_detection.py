from collections import defaultdict
import time

import numpy as np
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, f1_score

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from datasets import *

from deepod.models.dif import DeepIsolationForest
from deepod.models.dsvdd import DeepSVDD
from deepod.models.icl import ICL

from naf.autoencoder_model import AutoencoderModel
from naf.forests import ForestKind, TaskType
from naf.naf_model import NAFParams
from naf.naf_model_if import NeuralAttentionIsolationForest
from naf.naf_model import NeuralAttentionForest
from naf.naf_model_rf_multihead import NeuralMultiheadAttentionRandomForest
from logger.file_logger import FileLogger


def evaluate(y_true, scores):
    roc_auc = metrics.roc_auc_score(y_true, scores)
    ap = metrics.average_precision_score(y_true, scores)
    return roc_auc, ap


class AnomalyDetection:
    def __init__(self, num_seeds: int = 1, num_cross_val: int = 1, num_trees: int = 150, count_epoch: int = 300,
                 contaminations: int = 5):
        self._datasets = [  # default for new datasets - 6 dim size
            DatasetArrythmia(),  # 8 - 17 features
            DatasetCredit(),  # 12 - 30 features
            DatasetHaberman(),  # 2  - 3 features
            DatasetIonosphere(),  # 6 - 33 features
            DatasetPimaDiabetes(),  # 6 - 8 features
            DatasetSeismicBumps(),
            DatasetShuttle(),
            DatasetAnnhyroid(),
            DatasetBankAdditional(),
            DatasetCeleba()
        ] # PageBlocks,Annthyroid, KDD, Mulcross
        self._hidden_size = [8, 12, 6, 2, 2, 6, 3, 6, 6, 6]

        self._num_seeds = num_seeds
        self._num_cross_val = num_cross_val
        self._n_trees = num_trees
        self._count_epoch = count_epoch
        self._tree_type = ForestKind.RANDOM  # don't change
        self._contaminations = contaminations
        self._seed_variants = [1234 + 7 * i for i in range(self._num_seeds)]
        self._file_logger = FileLogger()
        self._file_logger.setup({"num_seeds": num_seeds,
                                 "num_cross_val": num_cross_val,
                                 "num_trees": num_trees,
                                 "count_epoch": self._count_epoch,
                                 "contaminations": self._contaminations})

        self._dict_models_naf = {
            "NAF-1-LAYER": NeuralAttentionForest,
            "NAF-3-LAYER": NeuralAttentionForest,
            "NAF-MH-3-HEAD-1-LAYER": NeuralMultiheadAttentionRandomForest,
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
                    # X_train, X_val, y_train, y_val = train_test_split(
                    #     dataset.X_train,
                    #     dataset.y_train,
                    #     test_size=0.2,
                    #     random_state=42)
                    # X_val = np.concatenate([X_train[y_train == 1], X_val])
                    # y_val = np.concatenate([y_train[y_train == 1], y_val])
                    start_time = time.time()
                    model.fit(dataset.X_train[dataset.y_train == 0])
                    t1 = time.time()

                    # default_predict = model.predict(X_val)
                    # if model_type == "IF":
                    #     default_predict[default_predict == 1] = 0
                    #     default_predict[default_predict == -1] = 1
                    # best_f1_score = f1_score(y_val, default_predict)
                    # if model_type == "IF":
                    #     best_thr = -model.offset_
                    # elif model_type == "DIF":
                    #     best_thr = model.threshold_
                    # elif model_type == "ICL":
                    #     best_thr = model.threshold_
                    # elif model_type == "DEEPSVD":
                    #     best_thr = model.threshold_
                    # for thr in np.linspace(best_thr - 0.15, best_thr + 0.15, num=10):
                    #     if model_type == "IF":
                    #         model.offset_ = -thr
                    #     elif model_type == "DIF":
                    #         model.threshold_ = thr
                    #     elif model_type == "ICL":
                    #         model.threshold_ = thr
                    #     elif model_type == "DEEPSVD":
                    #         model.threshold_ = thr
                    #     else:
                    #         raise Exception("CHECK THR OPTION")
                    #     y_preds = model.predict(X_val)
                    #     if model_type == "IF":
                    #         y_preds[y_preds == 1] = 0
                    #         y_preds[y_preds == -1] = 1
                    #     res = f1_score(y_val, y_preds)
                    #     if res > best_f1_score:
                    #         best_f1_score = res
                    #         if model_type == "IF":
                    #             best_thr = thr
                    #         else:
                    #             best_thr = thr
                    # if model_type == "IF":
                    #     model.offset_ = -best_thr
                    # else:
                    #     model.threshold_ = best_thr

                    y_preds = model.predict(dataset.X_test)
                    scores = model.decision_function(dataset.X_test)
                    if model_type == "IF":
                        y_preds[y_preds == 1] = 0
                        y_preds[y_preds == -1] = 1
                        scores = -scores
                    # dataset.plot_dataset(dataset.X_test, y_preds, "/Users/andreyageev/PycharmProjects/NAF/images/res.png")
                    all_F1.append(f1_score(dataset.y_test, y_preds))

                    auc, ap = evaluate(dataset.y_test, scores)
                    auc_lst.append(auc)
                    ap_lst.append(ap)
                    t1_lst.append(t1 - start_time)
                    # auc_lst[curr_run], ap_lst[curr_run] = auc, ap
                    # t1_lst[curr_run] = t1 - start_time
                    # curr_run += 1

                    # res = f1_score(y, prediction)
            self._file_logger.info("mean test f1: " + str(np.mean(np.array(all_F1))))
            avg_auc, avg_ap = np.average(auc_lst), np.average(ap_lst)
            std_auc, std_ap = np.std(auc_lst), np.std(ap_lst)
            avg_time = np.average(t1_lst)
            txt = f'avg_auc: {avg_auc:.4f}, std_auc: {std_auc:.4f}, ' \
                  f'avg_ap: {avg_ap:.4f}, std_ap: {std_ap:.4f}, ' \
                  f'avg_time: {avg_time:.1f}'
            self._file_logger.info(txt)
            # self._file_logger.end_logger("")

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
                    if model_type == "NAF-3-LAYER":
                        num_layers = 3
                    else:
                        num_layers = 1
                    hidden_size = dataset.X_train.shape[1] // 2
                    params = NAFParams(
                        kind=self._tree_type,
                        task=TaskType.REGRESSION,
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

                    # y_proba, x_recons, alphas, betas = model.predict(X_val,
                    #                                                  need_attention_weights=True)
                    # mse_values = np.mean(np.power(X_val - x_recons, 2), axis=1)
                    # scaler = MinMaxScaler(feature_range=(0, 1))
                    # mse_values = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))
                    # best_f1_score = 0
                    # best_thr = 0
                    # for thr in np.linspace(0, 0.2, num=100):
                    #     y_preds = np.where(mse_values < thr, 0, 1)
                    #     res = f1_score(y_val, y_preds)
                    #     if res > best_f1_score:
                    #         best_f1_score = res
                    #         best_thr = thr
                    # thr = best_thr
                    # if regularization_lambda != 0.0:
                    #     score = model.forest.decision_function(dataset.X_test.copy())
                    #     score = np.expand_dims(score, axis=1)
                    #     dataset_test = np.concatenate([dataset.X_test.copy().copy(), score], axis=1)
                    # else:
                    dataset_test = dataset.X_test

                    y_proba, x_recons, alphas, betas = model.predict(dataset_test,
                                                                     need_attention_weights=True)
                    mse_values = np.mean(np.power(dataset_test - x_recons, 2), axis=1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scores = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))
                    #
                    y_preds = np.where(scores < 0.1, 0, 1)
                    # dataset.plot_dataset(dataset.X_test, y_preds,
                    #                      "/Users/andreyageev/PycharmProjects/NAF/res_naf.png")
                    #
                    # mse_y0 = scores[dataset.y_test == 0]
                    # mse_y1 = scores[dataset.y_test == 1]
                    # plt.figure()
                    #
                    # plt.hist(mse_y0, bins=100, color='blue', alpha=0.8, label='normal')
                    # plt.hist(mse_y1, bins=100, color='red', alpha=0.8, label='anomaly')
                    # plt.legend(loc='upper right')
                    # plt.xlabel('MSE values')
                    # plt.ylabel('Frequency')
                    # plt.title(dataset.get_name())
                    # plt.show()
                    # plt.savefig(self._file_logger.filename + "_MSE_" + dataset.get_name() + ".png")

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
            # self._file_logger.end_logger("")

    def start_model_naf_impact_rf(self, model_type: str, regularization_lambda: float = 0.0):
        self._file_logger.info(model_type + ": reqularization lambda = " + str(regularization_lambda))
        model_cls = self._dict_models_naf[model_type]

        for dataset, hidden_size in zip(self._datasets, self._hidden_size):
            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")
            auc_lst, ap_lst = [[], [], [], []], [[], [], [], []]
            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    print(f"running: {k} and {seed}")
                    dataset.load()
                    forest_param = dict(
                        n_estimators=self._n_trees,
                        min_samples_leaf=1
                    )
                    if model_type == "NAF-3-LAYER":
                        num_layers = 3
                    else:
                        num_layers = 1
                    hidden_size = dataset.X_train.shape[1] // 2
                    params = NAFParams(
                        kind=self._tree_type,
                        task=TaskType.REGRESSION,
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
                    dataset.cross_validation_split(k)
                    Xn, Xa = dataset.split_normal_anomalous_train()
                    datasets = dataset.create_dataset_versions(Xn, Xa)
                    for idx, subset_dataset in enumerate(datasets):
                        X, Y = subset_dataset
                        model.fit(X, Y)
                        model.optimize_weights_unlabeled(Xn)

                        dataset_test = dataset.X_test

                        y_proba, x_recons, alphas, betas = model.predict(dataset_test,
                                                                         need_attention_weights=True)
                        mse_values = np.mean(np.power(dataset_test - x_recons, 2), axis=1)
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scores = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))

                        auc, ap = evaluate(dataset.y_test, scores)
                        print(f"auc = {auc}, ap = {ap}")
                        auc_lst[idx].append(auc)
                        ap_lst[idx].append(ap)
            for idx, subset_dataset in enumerate(datasets):
                avg_auc, avg_ap = np.average(auc_lst[idx]), np.average(ap_lst[idx])
                std_auc, std_ap = np.std(auc_lst[idx]), np.std(ap_lst[idx])
                txt = f'dataset_{idx} - avg_auc: {avg_auc:.4f}, std_auc: {std_auc:.4f}, ' \
                      f'avg_ap: {avg_ap:.4f}, std_ap: {std_ap:.4f}'
                self._file_logger.info(txt)

    def start_model_naf_injection(self, model_type: str, regularization_lambda: float = 0.0):
        self._file_logger.info(model_type + ": reqularization lambda = " + str(regularization_lambda))
        model_cls = self._dict_models_naf[model_type]

        for dataset, hidden_size in zip(self._datasets, self._hidden_size):
            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")
            auc_lst, ap_lst = [[], [], []], [[], [], []]
            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    print(f"running: {k} and {seed}")
                    dataset.load()
                    forest_param = dict(
                        n_estimators=self._n_trees,
                        min_samples_leaf=1
                    )
                    if model_type == "NAF-3-LAYER":
                        num_layers = 3
                    else:
                        num_layers = 1
                    hidden_size = dataset.X_train.shape[1] // 2
                    params = NAFParams(
                        kind=self._tree_type,
                        task=TaskType.REGRESSION,
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
                    dataset.cross_validation_split(k)
                    Xn, Xa = dataset.split_normal_anomalous_train()
                    datasets = dataset.create_dataset_versions(Xn, Xa, injection=True)
                    X_orig, Y_orig = dataset.create_dataset_versions(Xn, Xa, injection=False)[0]
                    for idx, subset_dataset in enumerate(datasets):
                        X, Y = subset_dataset
                        model.fit(X_orig, Y_orig)
                        model.optimize_weights_unlabeled(X)

                        dataset_test = dataset.X_test

                        y_proba, x_recons, alphas, betas = model.predict(dataset_test,
                                                                         need_attention_weights=True)
                        mse_values = np.mean(np.power(dataset_test - x_recons, 2), axis=1)
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scores = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))

                        auc, ap = evaluate(dataset.y_test, scores)
                        print(f"auc = {auc}, ap = {ap}")
                        auc_lst[idx].append(auc)
                        ap_lst[idx].append(ap)
            for idx, subset_dataset in enumerate(datasets):
                avg_auc, avg_ap = np.average(auc_lst[idx]), np.average(ap_lst[idx])
                std_auc, std_ap = np.std(auc_lst[idx]), np.std(ap_lst[idx])
                txt = f'dataset_{idx} - avg_auc: {avg_auc:.4f}, std_auc: {std_auc:.4f}, ' \
                      f'avg_ap: {avg_ap:.4f}, std_ap: {std_ap:.4f}'
                self._file_logger.info(txt)

    def get_contamination(self):
        self._file_logger.info("contamination")
        contaminations = self._contaminations
        model_names = ["NAF-IF", "NAF", "NAF-MH"]
        res_by_dataset = [[] for _ in range(len(self._datasets))]
        for dataset_id, (dataset, hidden_size) in enumerate(zip(self._datasets, self._hidden_size)):
            res_by_models = [[] for _ in range(len(model_names))]
            for model_id, model_type in enumerate(model_names):
                model_cls = self._dict_models_naf[model_type]
                for contamination_id, contamination_r in enumerate(np.linspace(1, 0.5, contaminations)):
                    auc_lst, ap_lst = [], []
                    t1_lst = []
                    self._file_logger.info(f"{dataset.get_name()} contamination rate={contamination_r}")
                    print(f"{dataset.get_name()}")
                    for k in range(0, self._num_cross_val):
                        for seed in self._seed_variants:
                            dataset.load()
                            dataset.cross_validation_split(k)
                            dataset.adjust_contamination(contamination_r)
                            # self._file_logger.info("contamination: " + str(dataset.get_contamiation()))
                            print(f"running: {k} and {seed} and {contamination_r}")
                            if model_type == "NAF":
                                forest_param = dict(
                                    n_estimators=self._n_trees,
                                    min_samples_leaf=1
                                )
                            else:
                                forest_param = dict(
                                    n_estimators=self._n_trees,
                                )
                            params = NAFParams(
                                kind=self._tree_type,
                                task=TaskType.REGRESSION,
                                mode='end_to_end',
                                n_epochs=self._count_epoch,
                                lr=0.01,
                                lam=1.0,
                                target_loss_weight=1.0,
                                hidden_size=hidden_size,
                                n_layers=1,
                                forest=forest_param,
                                random_state=seed
                            )
                            model = model_cls(params)

                            start_time = time.time()

                            model.fit(dataset.X_train[dataset.y_train == 0], dataset.y_train[dataset.y_train == 0])
                            model.optimize_weights_unlabeled(dataset.X_train[dataset.y_train == 0])
                            t1 = time.time()

                            y_proba, x_recons, alphas, betas = model.predict(dataset.X_test,
                                                                             need_attention_weights=True)
                            mse_values = np.mean(np.power(dataset.X_test - x_recons, 2), axis=1)
                            scaler = MinMaxScaler(feature_range=(0, 1))
                            scores = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))
                            auc, ap = evaluate(dataset.y_test, scores)
                            auc_lst.append(auc)
                            ap_lst.append(ap)
                            t1_lst.append(t1 - start_time)
                    res_by_models[model_id].append(np.average(auc_lst))
                    self._file_logger.info("auc_roc="+str(np.average(auc_lst)))
            res_by_dataset[dataset_id].append(res_by_models)
        # for dataset_id, dataset in enumerate(res_by_dataset):
            plt.figure()
            plt.title(self._datasets[dataset_id].get_name())
            for model_id, model in enumerate(res_by_dataset[dataset_id]):
                for i, line in enumerate(model):
                    plt.plot(np.linspace(1, 0.5, contaminations), line, linestyle='-', marker='*',
                             label=model_names[i])
            plt.ylim([0.0, 1.1])

            plt.xlabel("Fraction normal train data")
            plt.ylabel("AUC-ROC")
            plt.legend()

            plt.savefig(self._file_logger.filename + "_" + self._datasets[dataset_id].get_name()+".png")
