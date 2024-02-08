import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.basic_dataset import AbstractDataset

from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing


class DatasetHttp(AbstractDataset):
    def __init__(self):
        super().__init__()
        self._path = "/Users/andreyageev/PycharmProjects/ATIF/datasets/http.csv"

    def load(self):
        """https://www.openml.org/search?type=data&sort=runs&id=40897&status=active"""
        """https://github.com/dple/Datasets"""
        df = pd.read_csv(self._path)
        #
        # df.loc[df["Target"] == "Anomaly", "Target"] = 1
        # df.loc[df["Target"] == "Normal", "Target"] = 0
        # label_encoder = preprocessing.LabelEncoder()
        # df['Target'] = label_encoder.fit_transform(df['Target'])
        # # df = df.dropna(axis=0)
        # df['Target'] = df['Target'].map({0: 1, 1: 0})

        # df.loc[df["Target"] == 0, "Target"] = 1
        # df.loc[df["Target"] == 1, "Target"] = 0
        # df['Target'] = df['Target'].astype(int)
        # df['Target'] = df['Target'].map({'Anomaly': '1', 'Normal': '0'})
        # fraud = len(df[df['attack'] == 1])
        # valid = len(df[df['attack'] == 0])
        # fraud_data = df[df['attack'] == 1]
        # norm_data = df[df['attack'] == 0]
        fraud_data = df[df['attack'] == 1].iloc[0:50]
        norm_data = df[df['attack'] == 0].iloc[0:500]
        self.fraud_data = len(fraud_data)
        df = norm_data.append(fraud_data)
        ###########
        # import matplotlib.pyplot as plt
        # count_class = pd.value_counts(df['attack'])
        # count_class.plot(kind='bar', rot=0)
        # plt.title("Class Distribution")
        # plt.xticks(range(2), ["Normal", "Fraud"])
        # plt.xlabel("Class")
        # plt.ylabel("Frequency")
        # plt.savefig("/Users/andreyageev/PycharmProjects/ATIF/image/http_dataset.jpg")
        ###############
        # df['Target'] = df['Target'].str.replace(',', '')
        # df['Target'] = pd.to_numeric(df['Target'], errors='coerce')
        # df['Target'] = df['Target'].astype('str')
        # encoding = {"Anomaly": 1, "Normal": 0}
        # df['Target'].replace(encoding, inplace=True)
        data, labels = df.drop('attack', axis=1), df['attack']
        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)
        # col = data.columns
        # std = StandardScaler()
        # x = std.fit_transform(data)
        # data = pd.DataFrame(data=x, columns=col)
        # over = RandomOverSampler(random_state=42)
        # data, labels = over.fit_resample(data, labels)
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
        return "http"

