import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.basic_dataset import AbstractDataset

from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class DatasetArrythmia(AbstractDataset):
    def __init__(self):
        super().__init__()
        self._path = "/Users/andreyageev/PycharmProjects/ATIF/datasets/data_arrhythmia.csv"

    def load(self):
        """https://www.kaggle.com/code/mtavares51/binary-classification-on-arrhythmia-dataset"""
        target = "diagnosis"
        df = pd.read_csv(self._path, sep=';')
        df.dropna(axis=0, inplace=True)
        df.drop(df.columns[20:-2], axis=1, inplace=True)
        df.drop(['T', 'P', 'J', 'LG'], axis=1, inplace=True)
        j = []
        for i in df.diagnosis:
            if i in [3, 4, 5, 7, 8, 9, 14, 15]:
                j.append(1)
            else:
                j.append(0)
        df.diagnosis = j

        def label_encoding(old_column):
            le = LabelEncoder()
            le.fit(old_column)
            new_column = le.transform(old_column)
            return new_column

        # encoding string parameters
        for i in df.columns:
            if type(df[i][0]) == str:
                df[i] = label_encoding(df[i])
        df.astype(float)

        ###########
        # import matplotlib.pyplot as plt
        # count_class = pd.value_counts(df[target])
        # count_class.plot(kind='bar', rot=0)
        # plt.title("Class Distribution")
        # plt.xticks(range(2), ["Normal", "Fraud"])
        # plt.xlabel("Class")
        # plt.ylabel("Frequency")
        # plt.savefig("/Users/andreyageev/PycharmProjects/ATIF/arrythmia_dataset.jpg")
        ###############
        # extracting x and y
        labels = df[target].values
        fraud_data = df[df[target] == 1]
        norm_data = df[df[target] == 0]

        data = df.drop([target], axis=1).values
        data = data.astype(np.float32)

        self.data = data
        self.labels = labels

        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.33, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.y_train = y_train
        self.y_test = y_test

    # def cross_validation_split(self, k: int):
    #     X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.33, random_state=k)
    #     self.X_train = X_train
    #     self.X_test = X_test
    #     scaler = StandardScaler()
    #     scaler.fit(self.X_train)
    #     self.X_train = scaler.transform(self.X_train)
    #     self.X_test = scaler.transform(self.X_test)
    #     self.y_train = y_train
    #     self.y_test = y_test

    def plot_dataset(self, data, y, save_path):
        pass

    def get_name(self) -> str:
        return "arrythmia"
