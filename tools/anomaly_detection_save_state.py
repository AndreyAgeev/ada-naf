from collections import defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, f1_score

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from datasets.dataset_moon import DatasetMoon
from datasets.dataset_pima_diabetes import DatasetPimaDiabetes
from datasets.dataset_credit import DatasetCredit
from datasets.dataset_haberman import DatasetHaberman
from datasets.dataset_arrythmia import DatasetArrythmia
# from datasets.dataset_seismic_bumps import DatasetSeismicBumps
from datasets.dataset_http import DatasetHttp
from datasets.dataset_eeg_eye import DatasetEEGEye
from datasets.dataset_mulcross import DatasetMulcross
from datasets.dataset_ionosphere import DatasetIonosphere

from naf.forests import ForestKind, TaskType
from naf.naf_model import NAFParams
from naf.naf_model_if import NeuralAttentionIsolationForest
from naf.naf_model import NeuralAttentionForest
from naf.naf_model_rf_multihead import NeuralMultiheadAttentionIsolationForest
from logger.file_logger import FileLogger

# 1. Единые опции для NAF моделей
# 2, Единые опции в init для экспериментов
# 3. Отображение в output файлах опций запуска, плюс добавить уникальный временной ключ при создании файла
# 4. Добавить унифицированные опции для датасетов - трансформации одинаковые (StandartScaler, MinMaxScaler)
# 5. Проверить новые датасеты
# 6. Обновить опции NAFParams, чтобы можно было запускать разные MSE
# 7. Проверить MinMaxScaler - нормализация должна быть только на train и по тест потом брать
# 8. Добавить shuttle и pageblocks datasets
# 9. понять что за threshold и добавить его перебор по валидации
# 10. перейти только на нормальные данные при обучении
# 11. поменять в NAF разбиение датасета - чтобы один раз происходило


DICT_MODELS_NAF = {"NAF": NeuralAttentionForest,
                   "NAF-IF": NeuralAttentionIsolationForest,
                   "NAF-MH": NeuralMultiheadAttentionIsolationForest}


class AnomalyDetection:
    def __init__(self, num_seeds: int = 1, num_cross_val: int = 1, num_trees: int = 150):
        self._datasets = [
            # DatasetMoon(n_samples_normal=100, n_samples_outliners=50, rng=42)
            DatasetHaberman(),  # 2  - 3 features
            # DatasetArrythmia(),  # 8 - 17 features
            # DatasetCredit(),  # 12 - 30 features
            # DatasetPimaDiabetes(),  # 6 - 8 features
            # DatasetEEGEye(),  # 6 - 11 features (12 better)
            # DatasetMulcross(),  # 3 # super nice (6) - 4 features
            # DatasetIonosphere(),  # 6 - 33 features
            # DatasetHttp()  # 2 - 3 features
        ]
        # self._hidden_size = [6]
        self._hidden_size = [2, 8, 12, 6, 6, 3, 6, 2]

        # self._hidden_size = [2, 8, 12, 6, 6, 3, 6, 2]

        self._num_seeds = 1
        self._num_cross_val = 1
        self._n_trees = 150
        self._count_epoch = 300
        self._tree_type = ForestKind.RANDOM  # EXTRA лучше
        self._model_if = None
        self._model = None
        self.forest = None

        self._seed_variants = [1234 + 7 * i for i in range(self._num_seeds)]
        self._file_logger = FileLogger()
        self._file_logger.setup({"num_seeds": num_seeds,
                                 "num_cross_val": num_cross_val,
                                 "num_trees": num_trees,
                                 "count_epoch": self._count_epoch})

    def start_if(self):
        # self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")
        self._file_logger.info("IF:")
        self._file_logger.end_logger("")

        for dataset in self._datasets:
            # self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")

            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")
            all_F1 = []
            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    dataset.load()
                    forest_cls = IsolationForest
                    self.forest = forest_cls(n_estimators=self._n_trees)
                    self.forest.random_state = seed
                    dataset.cross_validation_split(k)
                    X_train, X_val_normal, y_train, y_val_normal = train_test_split(
                        dataset.X_train[dataset.y_train == 0],
                        dataset.y_train[dataset.y_train == 0],
                        test_size=0.2,
                        random_state=42)
                    # X_train, X_val, y_train, y_val = train_test_split(dataset.X_train, dataset.y_train, test_size=0.2,
                    #                                                   random_state=42)
                    self.forest.fit(X_train, y_train)
                    # dataset.plot_dataset(dataset.X_train, dataset.y_train, "/Users/andreyageev/PycharmProjects/NAF/images/train.png")
                    # dataset.plot_dataset(dataset.X_test, dataset.y_test, "/Users/andreyageev/PycharmProjects/NAF/images/test.png")

                    # best_f1_score = 0
                    # best_thr = 0
                    # for thr in np.linspace(0.4, 0.6, num=100):
                    #     self.forest.offset_ = -thr
                    #     y_preds = self.forest.predict(X_val)
                    #     y_preds[y_preds == 1] = 0
                    #     y_preds[y_preds == -1] = 1
                    #     res = f1_score(y_val, y_preds)
                    #     if res > best_f1_score:
                    #         best_f1_score = res
                    #         best_thr = -thr
                    # self.forest.offset_ = best_thr
                    y_preds = self.forest.predict(dataset.X_test)

                    y_preds[y_preds == 1] = 0
                    y_preds[y_preds == -1] = 1
                    # dataset.plot_dataset(dataset.X_test, y_preds, "/Users/andreyageev/PycharmProjects/NAF/images/res.png")
                    all_F1.append(f1_score(dataset.y_test, y_preds))
                    # print(f1_score(dataset.y_test, y_preds))
            self._file_logger.info("mean test score = " + str(np.mean(np.array(all_F1))))
            self._file_logger.end_logger("")

    def start_model_naf(self, model_type: str):
        self._file_logger.info(model_type)
        model_cls = DICT_MODELS_NAF[model_type]

        for dataset, hidden_size in zip(self._datasets, self._hidden_size):
            all_F1 = []
            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")

            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    dataset.load()
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
                        forest=dict(
                            n_estimators=self._n_trees,
                            min_samples_leaf=1
                        ),
                        random_state=seed
                    )
                    self._model = model_cls(params)
                    dataset.load()
                    dataset.cross_validation_split(k)
                    # X_train, X_val_normal, y_train, y_val_normal = train_test_split(
                    #     dataset.X_train[dataset.y_train == 0],
                    #     dataset.y_train[dataset.y_train == 0],
                    #     test_size=0.2,
                    #     random_state=42)
                    # self._model.fit(X_train, y_train)
                    # self._model.optimize_weights_unlabeled(
                    #     dataset.X_train[dataset.y_train == 0])  # здесь разбиение само произойдет такое же
                    #
                    # _, X_val_anomaly, _, y_val_anomaly = train_test_split(dataset.X_train[dataset.y_train == 1],
                    #                                                       dataset.y_train[dataset.y_train == 1],
                    #                                                       test_size=0.2,
                    #                                                       random_state=42)
                    # X_val = np.concatenate([X_val_normal, X_val_anomaly])
                    # y_val = np.concatenate([y_val_normal, y_val_anomaly])
                    X_train, X_val, y_train, y_val = train_test_split(
                        dataset.X_train,
                        dataset.y_train,
                        test_size=0.2,
                        random_state=42)
                    X_val = np.concatenate([X_train[y_train == 1], X_val])
                    y_val = np.concatenate([y_train[y_train == 1], y_val])
                    self._model.fit(X_train[y_train == 0], y_train[y_train == 0])
                    self._model.optimize_weights_unlabeled(X_train[y_train == 0]) # еще и тут разбить??
                    y_proba, x_recons, alphas, betas = self._model.predict(X_val,
                                                                           need_attention_weights=True)
                    mse_values = np.mean(np.power(X_val - x_recons, 2), axis=1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    mse_values = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))

                    best_f1_score = 0
                    best_thr = 0
                    for thr in np.linspace(0, 0.2, num=100):  # для IF такой же перебор надо устроить
                        y_preds = np.where(mse_values < thr, 0, 1)
                        res = f1_score(y_val, y_preds)
                        if res > best_f1_score:
                            best_f1_score = res
                            best_thr = thr
                    thr = best_thr

                    y_proba, x_recons, alphas, betas = self._model.predict(dataset.X_test,
                                                                           need_attention_weights=True)
                    mse_values = np.mean(np.power(dataset.X_test - x_recons, 2), axis=1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    mse_values = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))

                    y_preds = np.where(mse_values < thr, 0, 1)
                    # dataset.plot_dataset(dataset.X_test, y_preds,
                    #                      "/Users/andreyageev/PycharmProjects/NAF/images/res_naf.png")
                    #
                    mse_y0 = mse_values[dataset.y_test == 0]
                    mse_y1 = mse_values[dataset.y_test == 1]
                    plt.hist(mse_y0, bins=100, color='blue', alpha=0.5, label='normal')
                    plt.hist(mse_y1, bins=100, color='red', alpha=0.5, label='anomaly')
                    plt.legend(loc='upper right')
                    plt.xlabel('MSE values')
                    plt.ylabel('Frequency')
                    plt.show()
                    print(f1_score(dataset.y_test, y_preds))
                    all_F1.append(f1_score(dataset.y_test, y_preds))
            self._file_logger.info("mean test score = " + str(np.mean(np.array(all_F1))))
            self._file_logger.end_logger("")

    def start_model_if(self):
        self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")
        self._file_logger.info("NAF IF:")
        self._file_logger.end_logger("")
        for dataset, hidden_size in zip(self._datasets, self._hidden_size):
            self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")
            all_F1 = []
            self._file_logger.info(f"{dataset.get_name()}")
            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    params_if = NAFParams(  # потом просто тут менять параметры и заново создавать в методах start
                        kind=self._tree_type,
                        task=TaskType.REGRESSION,  # не важно, мы же обучаем без меток
                        mode='end_to_end',
                        n_epochs=self._count_epoch,
                        lr=0.01,
                        lam=1.0,
                        target_loss_weight=1.0,
                        hidden_size=hidden_size,
                        n_layers=1,
                        forest=dict(
                            n_estimators=self._n_trees,
                        ),
                        random_state=seed
                    )
                    self._model_if = NeuralAttentionIsolationForest(params_if)

                    dataset.load()
                    dataset.cross_validation_split(k)
                    X_train, X_val_normal, y_train, y_val_normal = train_test_split(
                        dataset.X_train[dataset.y_train == 0],
                        dataset.y_train[dataset.y_train == 0],
                        test_size=0.2,
                        random_state=42)
                    self._model.fit(X_train, y_train)
                    self._model.optimize_weights_unlabeled(
                        dataset.X_train[dataset.y_train == 0])  # здесь разбиение само произойдет такое же

                    _, X_val_anomaly, _, y_val_anomaly = train_test_split(dataset.X_train[dataset.y_train == 1],
                                                                          dataset.y_train[dataset.y_train == 1],
                                                                          test_size=0.2,
                                                                          random_state=42)
                    X_val = np.concatenate([X_val_normal, X_val_anomaly])
                    y_val = np.concatenate([y_val_normal, y_val_anomaly])

                    y_proba, x_recons, alphas, betas = self._model.predict(X_val,
                                                                           need_attention_weights=True)
                    mse_values = np.mean(np.power(X_val - x_recons, 2), axis=1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    mse_values = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))

                    best_f1_score = 0
                    best_thr = 0
                    for thr in np.linspace(0, 0.2, num=100):  # для IF такой же перебор надо устроить
                        y_preds = np.where(mse_values < thr, 0, 1)
                        res = f1_score(y_val, y_preds)
                        if res > best_f1_score:
                            best_f1_score = res
                            best_thr = thr
                    thr = best_thr

                    y_proba, x_recons, alphas, betas = self._model.predict(dataset.X_test,
                                                                           need_attention_weights=True)
                    mse_values = np.mean(np.power(dataset.X_test - x_recons, 2), axis=1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    mse_values = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))

                    y_preds = np.where(mse_values < thr, 0, 1)
                    # dataset.plot_dataset(dataset.X_test, y_preds,
                    #                      "/Users/andreyageev/PycharmProjects/NAF/images/res_naf.png")
                    #
                    # mse_y0 = mse_values[dataset.y_test == 0]
                    # mse_y1 = mse_values[dataset.y_test == 1]
                    # plt.hist(mse_y0, bins=100, color='blue', alpha=0.5, label='normal')
                    # plt.hist(mse_y1, bins=100, color='red', alpha=0.5, label='anomaly')
                    # plt.legend(loc='upper right')
                    # plt.xlabel('MSE values')
                    # plt.ylabel('Frequency')
                    # plt.show()
                    all_F1.append(f1_score(dataset.y_test, y_preds))
            self._file_logger.info("mean test score = " + str(np.mean(np.array(all_F1))))
            self._file_logger.end_logger("")

    def start_model_naf_multihead(self):
        self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")
        self._file_logger.info("NAF-MH:")
        self._file_logger.end_logger("")

        for dataset, hidden_size in zip(self._datasets, self._hidden_size):
            self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")

            all_F1 = []
            self._file_logger.info(f"{dataset.get_name()}")

            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    params = NAFParams(  # потом просто тут менять параметры и заново создавать в методах start
                        kind=self._tree_type,
                        task=TaskType.REGRESSION,  # не важно, мы же обучаем без меток
                        mode='end_to_end',
                        n_epochs=self._count_epoch,
                        lr=0.01,
                        lam=1.0,
                        target_loss_weight=1.0,
                        hidden_size=hidden_size,
                        n_layers=1,
                        forest=dict(
                            n_estimators=self._n_trees,
                        ),
                        random_state=seed
                    )
                    self._model = NeuralMultiheadAttentionIsolationForest(params)

                    dataset.load()
                    dataset.cross_validation_split(k)
                    X_train, X_val_normal, y_train, y_val_normal = train_test_split(
                        dataset.X_train[dataset.y_train == 0],
                        dataset.y_train[dataset.y_train == 0],
                        test_size=0.2,
                        random_state=42)
                    self._model.fit(X_train, y_train)
                    self._model.optimize_weights_unlabeled(
                        dataset.X_train[dataset.y_train == 0])  # здесь разбиение само произойдет такое же

                    _, X_val_anomaly, _, y_val_anomaly = train_test_split(dataset.X_train[dataset.y_train == 1],
                                                                          dataset.y_train[dataset.y_train == 1],
                                                                          test_size=0.2,
                                                                          random_state=42)
                    X_val = np.concatenate([X_val_normal, X_val_anomaly])
                    y_val = np.concatenate([y_val_normal, y_val_anomaly])

                    y_proba, x_recons, alphas, betas = self._model.predict(X_val,
                                                                           need_attention_weights=True)
                    mse_values = np.mean(np.power(X_val - x_recons, 2), axis=1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    mse_values = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))

                    best_f1_score = 0
                    best_thr = 0
                    for thr in np.linspace(0, 0.2, num=100):  # для IF такой же перебор надо устроить
                        y_preds = np.where(mse_values < thr, 0, 1)
                        res = f1_score(y_val, y_preds)
                        if res > best_f1_score:
                            best_f1_score = res
                            best_thr = thr
                    thr = best_thr

                    y_proba, x_recons, alphas, betas = self._model.predict(dataset.X_test,
                                                                           need_attention_weights=True)
                    mse_values = np.mean(np.power(dataset.X_test - x_recons, 2), axis=1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    mse_values = scaler.fit_transform(np.array(mse_values).reshape(-1, 1))

                    y_preds = np.where(mse_values < thr, 0, 1)
                    # dataset.plot_dataset(dataset.X_test, y_preds,
                    #                      "/Users/andreyageev/PycharmProjects/NAF/images/res_naf.png")
                    #
                    # mse_y0 = mse_values[dataset.y_test == 0]
                    # mse_y1 = mse_values[dataset.y_test == 1]
                    # plt.hist(mse_y0, bins=100, color='blue', alpha=0.5, label='normal')
                    # plt.hist(mse_y1, bins=100, color='red', alpha=0.5, label='anomaly')
                    # plt.legend(loc='upper right')
                    # plt.xlabel('MSE values')
                    # plt.ylabel('Frequency')
                    # plt.show()
                    # print(f1_score(dataset.y_test, y_preds))
                    all_F1.append(f1_score(dataset.y_test, y_preds))
            self._file_logger.info("mean test score = " + str(np.mean(np.array(all_F1))))
            self._file_logger.end_logger("")

    def start_model_deepod(self):
        self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")
        self._file_logger.info("DEEP IF:")
        self._file_logger.end_logger("")

        for dataset, hidden_size in zip(self._datasets, self._hidden_size):
            all_F1 = []
            self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")
            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")

            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    dataset.load()
                    from deepod.models.dif import DeepIsolationForest
                    model_configs = {'epochs': 300,
                                     'n_ensemble': 50,
                                     'n_estimators': 6,
                                     "random_state": seed}
                    self._model = DeepIsolationForest(**model_configs)

                    dataset.load()
                    dataset.cross_validation_split(k)
                    X_train, X_val, y_train, y_val = train_test_split(
                        dataset.X_train,
                        dataset.y_train,
                        test_size=0.2,
                        random_state=42)
                    X_val = np.concatenate([X_train[y_train == 1], X_val])
                    y_val = np.concatenate([y_train[y_train == 1], y_val])
                    self._model.fit(X_train[y_train == 0], y_train[y_train == 0])
                    best_f1_score = 0
                    best_thr = 0
                    for thr in np.linspace(0.2, 0.5, num=100):
                        self._model.treshold_ = thr
                        y_preds = self._model.predict(X_val)
                        res = f1_score(y_val, y_preds)
                        if res > best_f1_score:
                            best_f1_score = res
                            best_thr = thr
                    self._model.threshold_ = best_thr
                    y_preds = self._model.predict(dataset.X_test)
                    all_F1.append(f1_score(dataset.y_test, y_preds))
                    print(f1_score(dataset.y_test, y_preds))
            self._file_logger.info("mean test score = " + str(np.mean(np.array(all_F1))))
            self._file_logger.end_logger("")

    def start_model_deepsvdd(self):
        self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")
        self._file_logger.info("DEEP SVDD:")
        self._file_logger.end_logger("")

        for dataset, hidden_size in zip(self._datasets, self._hidden_size):
            all_F1 = []
            self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")
            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")

            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    dataset.load()
                    from deepod.models.dsvdd import DeepSVDD
                    self._model = DeepSVDD(epochs=self._count_epoch, random_state=seed, device="cpu")
                    dataset.load()
                    dataset.cross_validation_split(k)
                    X_train, _, y_train, _ = train_test_split(
                        dataset.X_train,
                        dataset.y_train,
                        test_size=0.2,
                        random_state=42)
                    self._model.fit(X_train, y_train)
                    y_preds = self._model.predict(dataset.X_test)
                    print(f1_score(dataset.y_test, y_preds))
                    all_F1.append(f1_score(dataset.y_test, y_preds))
            self._file_logger.info("mean test score = " + str(np.mean(np.array(all_F1))))
            self._file_logger.end_logger("")

    def start_model_icl(self):
        self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")
        self._file_logger.info("ICL:")
        self._file_logger.end_logger("")

        for dataset, hidden_size in zip(self._datasets, self._hidden_size):
            all_F1 = []
            self._file_logger.setup("/Users/andreyageev/PycharmProjects/NAF/output.txt")
            self._file_logger.info(f"{dataset.get_name()}")
            print(f"{dataset.get_name()}")

            for k in range(0, self._num_cross_val):
                for seed in self._seed_variants:
                    dataset.load()
                    from deepod.models.icl import ICL
                    self._model = ICL(epochs=self._count_epoch, random_state=seed, device="cpu")
                    dataset.load()
                    dataset.cross_validation_split(k)
                    X_train, _, y_train, _ = train_test_split(
                        dataset.X_train,
                        dataset.y_train,
                        test_size=0.2,
                        random_state=42)
                    self._model.fit(X_train, y_train)
                    y_preds = self._model.predict(dataset.X_test)
                    print(f1_score(dataset.y_test, y_preds))
                    all_F1.append(f1_score(dataset.y_test, y_preds))
            self._file_logger.info("mean test score = " + str(np.mean(np.array(all_F1))))
            self._file_logger.end_logger("")