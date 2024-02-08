import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from datasets.basic_dataset import AbstractDataset


class DatasetEEGEye(AbstractDataset):
    def __init__(self):
        super().__init__()

    def preprocess(self, df):
        # тут может быть не так деление на трейн тест происходит и скалировать не надо
        pass


    def load(self):
        """https://www.kaggle.com/datasets/robikscube/eye-state-classification-eeg-dataset"""
        """https://www.kaggle.com/code/parhammostame/eeg-eye-state-classification-using-kernel-svm"""
        df = pd.read_csv("/Users/andreyageev/PycharmProjects/AttentionBasedIsolationForestScoreFunction/datasets/EEG_Eye_State_Classification.csv")
        # define sampling rate, time vector, and electrode list (columns list)
        df = df.iloc[0:1500]
        Fs = 128  # (number of samples / 117s length of data mentioned on the data description) rounded to the closest integer.
        t = np.arange(0, len(df) * 1 / Fs, 1 / Fs)
        cols = df.columns.tolist()[:-1]
        Y = df['eyeDetection']
        X = df.drop(columns='eyeDetection')
        # Find outliers and put Nan instead
        X = X.apply(stats.zscore, axis=0)
        X = X.applymap(lambda x: np.nan if (abs(x) > 4) else x)

        # recalculate outliers with ignoring nans since the first calculation was biased with the huge outliers!
        X = X.apply(stats.zscore, nan_policy='omit', axis=0)
        X = X.applymap(lambda x: np.nan if (abs(x) > 4) else x)
        from scipy import signal, interpolate

        def interp(x):
            t_temp = t[x.index[~x.isnull()]]
            x = x[x.index[~x.isnull()]]
            clf = interpolate.interp1d(t_temp, x, kind='cubic')
            return clf(t)

        # interpolate the nans using cubic spline method
        X_interp = X.apply(interp, axis=0)
        from sklearn.decomposition import FastICA

        # apply ICA to drop non-electrophysiolgoical components (requires familiarity with EEG data)
        ica = FastICA(max_iter=2000, random_state=0)
        X_pcs = pd.DataFrame(ica.fit_transform(X_interp))
        X_pcs.columns = ['PC' + str(ind + 1) for ind in range(X_pcs.shape[-1])]
        X_pcs = X_pcs.drop(columns=['PC1', 'PC7'])

        # reconstruct clean EEG after dropping the bad components
        ica.mixing_ = np.delete(ica.mixing_, [0, 6], axis=1)
        X_interp_clean = pd.DataFrame(ica.inverse_transform(X_pcs))
        X_interp_clean.columns = cols

        # now that data is clean, extract alpha waves magnitude from the clean signals

        # filter the data between 8-12 Hz (note that data has been rescaled to original scale after filtering for comparable visualization)
        b, a = signal.butter(6, [8 / Fs * 2, 12 / Fs * 2], btype='bandpass')
        X_interp_clean_alpha = X_interp_clean.apply(
            lambda x: signal.filtfilt(b, a, x) / max(abs(signal.filtfilt(b, a, x))) * max(abs(x)), axis=0)

        # extract envelope of the Alpha waves
        X_interp_clean_alpha = X_interp_clean_alpha.apply(lambda x: np.abs(signal.hilbert(x)), axis=0)
        X_interp_clean_alpha.columns = cols
        # drop features with high correlations
        X = X_interp_clean_alpha
        Cols_corr = X.corr()
        # exclude columns with high correlation
        cols_drop_ind = [0] * len(cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                if (i < j) & abs(Cols_corr.iloc[i, j] >= 0.8):
                    cols_drop_ind[j] = 1

        cols_drop = [cols[ind] for ind in range(len(cols_drop_ind)) if cols_drop_ind[ind]]
        X.drop(columns=cols_drop, inplace=True)
        # scaler = StandardScaler()
        # data = scaler.fit_transform(X)
        self.data = X
        self.labels = Y

        counts = pd.value_counts(Y)

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
        return "eeg_eye"

