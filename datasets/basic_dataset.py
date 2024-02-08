from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class AbstractDataset(ABC):
    """Abstract class for dataset."""
    def __init__(self):
        self.data = None
        self.labels = None
        self.X_train: np.ndarray = []
        self.X_test: np.ndarray = []
        self.y_train: np.ndarray = []
        self.y_test: np.ndarray = []

        self._curr_contamination = None

    # @abstractmethod
    # def cross_validation_split(self, k: int):
    #     pass

    @abstractmethod
    def load(self):
        """Process function."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def cross_validation_split(self, k: int):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.33,
                                                            random_state=k, stratify=self.labels)
        np.random.seed(k)
        self.seed = k
        if hasattr(X_train, "values"):
            self.X_train = X_train.values
            self.X_test = X_test.values
        else:
            self.X_train = X_train
            self.X_test = X_test
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        if hasattr(y_train, "values") or isinstance(y_train, pd.Series):
            self.y_train = y_train.values
            self.y_test = y_test.values
        else:
            self.y_train = y_train
            self.y_test = y_test

    def split_normal_anomalous_train(self):
        Xn = self.X_train[self.y_train == 0]
        Xa = self.X_train[self.y_train == 1]
        return Xn, Xa

    # def create_dataset_versions(self, Xn, Xa):
    #     X1 = np.concatenate((0.7 * Xn, 0.3 * Xa))
    #     X2 = np.concatenate((0.5 * Xn, 0.5 * Xa))
    #     X3 = np.concatenate((0.3 * Xn, 0.7 * Xa))
    #     return X1, X2, X3

    # def create_dataset_versions(self, Xn, Xa):
    #     # Create label arrays
    #     labels_n = np.zeros(len(Xn))
    #     labels_a = np.ones(len(Xa))
    #
    #     # Calculate the number of samples from each set for the mix
    #     num_n_for_X1 = int(0.7 * len(Xn))
    #     num_a_for_X1 = int(0.3 * len(Xa))
    #     num_n_for_X2 = int(0.5 * len(Xn))
    #     num_a_for_X2 = int(0.5 * len(Xa))
    #     num_n_for_X3 = int(0.3 * len(Xn))
    #     num_a_for_X3 = int(0.7 * len(Xa))
    #
    #     # Create datasets
    #     X1 = np.concatenate((Xn[:num_n_for_X1], Xa[:num_a_for_X1]))
    #     X2 = np.concatenate((Xn[:num_n_for_X2], Xa[:num_a_for_X2]))
    #     X3 = np.concatenate((Xn[:num_n_for_X3], Xa[:num_a_for_X3]))
    #
    #     # Create corresponding label arrays
    #     Y1 = np.concatenate((labels_n[:num_n_for_X1], labels_a[:num_a_for_X1]))
    #     Y2 = np.concatenate((labels_n[:num_n_for_X2], labels_a[:num_a_for_X2]))
    #     Y3 = np.concatenate((labels_n[:num_n_for_X3], labels_a[:num_a_for_X3]))
    #
    #     return (X1, Y1), (X2, Y2), (X3, Y3)

    import numpy as np

    def create_dataset_versions(self, Xn, Xa, injection=False):
        # Create label arrays
        labels_n = np.zeros(len(Xn))
        labels_a = np.ones(len(Xa))

        num_anomalous = len(Xa)
        total = 2 * num_anomalous
        if injection is False:
            proportions = [1.0, 0.7, 0.5, 0.3]
        else:
            proportions = [0.0, 0.05, 0.1]

        datasets = []
        np.random.seed(self.seed)

        for proportion_a in proportions:
            # Вычисляем количество нормальных и аномальных сэмплов в соответствии с пропорцией
            num_anomalous_current = int(proportion_a * num_anomalous)
            num_normal = total - num_anomalous_current

            # Убедимся, что не превышаем количество доступных нормальных сэмплов
            num_normal = min(num_normal, len(Xn))

            # Конкатенация данных и меток
            X = np.concatenate((Xn[:num_normal], Xa[:num_anomalous_current]))
            Y = np.concatenate((labels_n[:num_normal], labels_a[:num_anomalous_current]))

            # Перемешиваем данные и метки
            combined = list(zip(X, Y))
            np.random.shuffle(combined)
            X_shuffled, Y_shuffled = zip(*combined)

            # Преобразование обратно в numpy массивы
            X_shuffled = np.array(X_shuffled)
            Y_shuffled = np.array(Y_shuffled)
            print(len(Y_shuffled[Y_shuffled == 1]))

            datasets.append((X_shuffled, Y_shuffled))

        return datasets

    def create_noisy_versions(self, Xn, Xa):
        Xn1 = np.concatenate((0.9 * Xn, 0.1 * Xa))
        Xn2 = np.concatenate((0.95 * Xn, 0.05 * Xa))
        return Xn1, Xn2

    def random_subset_split(self, X):
        # Определяем размеры для Xt1, Xt2, и Xt3 (70%, 65%, 60%)
        np.random.seed(self.seed)
        sizes = [0.7, 0.7, 0.7]

        # Создаем подмножества
        Xt1 = np.random.choice(X, size=int(len(X) * sizes[0]), replace=False)
        Xt2 = np.random.choice(X, size=int(len(X) * sizes[1]), replace=False)
        Xt3 = np.random.choice(X, size=int(len(X) * sizes[2]), replace=False)

        return Xt1, Xt2, Xt3

    def get_contamiation(self):
        return f"dataset={self.data[self.labels==1].shape[0] / self.data.shape[0]}, " \
               f"train={self.X_train[self.y_train==1].shape[0] / self.X_train.shape[0]}, " \
               f"test={self.X_test[self.y_test==1].shape[0] / self.X_test.shape[0]} " \
               f"train_changed = {len(self.X_train[self.y_train==0]) / self.basic_x_train_size} "

    def adjust_contamination(self, contamination_r, swap_ratio=0.05, random_state=42):
        """
        add anomalies to training data to replicate anomaly contaminated data sets.
        we randomly swap 5% features of two anomalies to avoid duplicate contaminated anomalies.
        """
        rng = np.random.RandomState(random_state)

        norm_idx = np.where(self.y_train == 0)[0]
        self.basic_x_train_size = len(norm_idx)
        n_remove_norm = int(len(norm_idx) * (1 - contamination_r))
        remove_id_norm = rng.choice(norm_idx, n_remove_norm, replace=False)
        if isinstance(self.X_train, np.ndarray):
            self.X_train = np.delete(self.X_train, remove_id_norm, 0)
            self.y_train = np.delete(self.y_train, remove_id_norm, 0)
        else:
            self.X_train = self.data.drop(self.X_train.index[remove_id_norm])
            self.y_train = self.labels.drop(self.y_train.index[remove_id_norm])
