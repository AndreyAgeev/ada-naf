import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from datasets.basic_dataset import AbstractDataset


class DatasetKDD2014(AbstractDataset):
    def __init__(self):
        super().__init__()

    def load(self):
        """https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/tree/main/numerical%20data/DevNet%20datasets
        https://github.com/GuansongPang/deviation-network"""
        data = pd.read_csv("/Users/andreyageev/PycharmProjects/NAF/dataset_folder/KDD2014_donors_10feat_nomissing_normalised.csv")
        data = data.astype(float)
        fraud_data = data[data['class'] == 1].iloc[0:50]
        norm_data = data[data['class'] == 0].iloc[0:500]

        data = norm_data.append(fraud_data)
        data = data.astype(float)
        data, labels = data.drop("class", axis=1), data["class"]
        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)
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

    def plot_dataset(self, data, y, save_path):
        pass

    def get_name(self) -> str:
        return "kdd2014"

