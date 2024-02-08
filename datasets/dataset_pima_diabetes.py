import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .basic_dataset import AbstractDataset


class DatasetPimaDiabetes(AbstractDataset):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.data = None
        self._path = "/Users/andreyageev/PycharmProjects/ATIF/datasets/diabetes.csv"

    def load(self):
        """https://www.kaggle.com/code/hafizramadan/data-science-project-iii"""
        df = pd.read_csv(self._path)
        fraud = len(df[df['Outcome'] == 1])
        valid = len(df[df['Outcome'] == 0])
        data, labels = df.drop('Outcome', axis=1), df['Outcome']
        # col = data.columns
        # std = StandardScaler()
        # data = std.fit_transform(data)
        # data = pd.DataFrame(data=x, columns=col)
        self.data = data
        self.labels = labels
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.33, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.y_train = y_train.values
        self.y_test = y_test.values

    # def cross_validation_split(self, k: int):
    #     X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.33, random_state=k)
    #     self.X_train = X_train
    #     self.X_test = X_test
    #     scaler = StandardScaler()
    #     scaler.fit(self.X_train)
    #     self.X_train = scaler.transform(self.X_train)
    #     self.X_test = scaler.transform(self.X_test)
    #     self.y_train = y_train.values
    #     self.y_test = y_test.values

    def get_name(self) -> str:
        return "diabetes"
