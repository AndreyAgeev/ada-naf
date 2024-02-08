import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from datasets.basic_dataset import AbstractDataset


class DatasetSeismicBumps(AbstractDataset):
    def __init__(self):
        super().__init__()

    def load(self):
        """https://www.kaggle.com/datasets/pranabroy94/seismic-bumps-data-set"""
        df_sb = pd.read_csv("/Users/andreyageev/PycharmProjects/AttentionBasedIsolationForestScoreFunction/datasets/seismic-bumps.csv")

        # Get one hot encoding of categorical columns
        cat_cols = ['seismic', 'seismoacoustic', 'shift', 'hazard']
        df_sb_1h = pd.get_dummies(df_sb[cat_cols])

        # Drop category columns as they are now encoded
        df_sb = df_sb.drop(cat_cols, axis=1)

        # Combine hot encoded and non-hot encoded columns
        df_sb = pd.concat([df_sb, df_sb_1h], axis=1)

        # Perform a min-max scaling as a normalization
        # If this were a model that would get data outside of this dataset, we'd need to save these min/max values for future conversion use
        # df_sb = (df_sb - df_sb.min()) / (df_sb.max() - df_sb.min())

        # Upon reviewing the data, we do not have any nbumps6; dropping
        df_sb = df_sb.drop(['nbumps6', 'nbumps7', 'nbumps89'], axis=1)

        # Move class column to end:
        df_sb.insert(len(df_sb.columns) - 1, 'class', df_sb.pop('class'))

        # df_sb = shuffle(df_sb)
        data, labels = df_sb.drop("class", axis=1), df_sb["class"]
        fraud = labels[labels==1]
        normal = labels[labels==0]

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
        return "seismic_bumps"
