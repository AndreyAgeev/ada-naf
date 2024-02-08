import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .basic_dataset import AbstractDataset
from sklearn import preprocessing


class DatasetCredit(AbstractDataset):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.data = None
        self._path = "/Users/andreyageev/PycharmProjects/ATIF/datasets/creditcard.csv"
        self._target_col = "Class"

    def load(self):
        """https://www.kaggle.com/code/shivamsekra/credit-card-fraud-detection-eda-isolation-forest"""
        data = pd.read_csv(self._path)
        label_encoder = preprocessing.LabelEncoder()
        data[self._target_col] = label_encoder.fit_transform(data[self._target_col])
        # fraud_data = data[data['Class'] == 1]
        # norm_data = data[data['Class'] == 0]
        fraud_data = data[data['Class'] == 1].iloc[0:400]
        self.fraud_data = len(fraud_data)
        norm_data = data[data['Class'] == 0].iloc[0:1500]
        data = norm_data.append(fraud_data)

        data, labels = data.drop(self._target_col, axis=1), data[self._target_col]
        from sklearn.preprocessing import RobustScaler

        rob_scaler = RobustScaler()
        data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
        data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1, 1))
        data.drop(['Time', 'Amount'], axis=1, inplace=True)
        scaled_amount = data['scaled_amount']
        scaled_time = data['scaled_time']
        data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        data.insert(0, 'scaled_amount', scaled_amount)
        data.insert(1, 'scaled_time', scaled_time)

        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)
        ##########
        # col = data.columns
        # std = StandardScaler()
        # x = std.fit_transform(data)
        # data = pd.DataFrame(data=x, columns=col)
        #####################
        self.data = data
        self.labels = labels
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.33, random_state=42)
        self.X_train = X_train.values
        self.X_test = X_test.values
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.y_train = y_train.values
        self.y_test = y_test.values

    # def cross_validation_split(self, k: int):
    #     X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.33, random_state=k)
    #     self.X_train = X_train.values
    #     self.X_test = X_test.values
    #     scaler = StandardScaler()
    #     scaler.fit(self.X_train)
    #     self.X_train = scaler.transform(self.X_train)
    #     self.X_test = scaler.transform(self.X_test)
    #     self.y_train = y_train.values
    #     self.y_test = y_test.values

    def plot_dataset(self, data, y, save_path):
        pass

    def get_name(self) -> str:
        return "credit"

