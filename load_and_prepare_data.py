# load_and_prepare_data.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, time_steps=12, scaler_x=MinMaxScaler(), scaler_y=MinMaxScaler()):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.time_steps = time_steps
        x_scaled = scaler_x.fit_transform(x)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        self.x, self.y = [], []
        for i in range(len(x_scaled) - time_steps):
            self.x.append(x_scaled[i:(i + time_steps)])
            self.y.append(y_scaled[i + time_steps])
        self.x = torch.FloatTensor(np.array(self.x)) # need inverse transform
        self.y = torch.FloatTensor(np.array(self.y)) # need inverse transform
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)
    
    def dimension(self):
        return self.x.shape[2]
    
    def scaler(self):
        return self.scaler_x, self.scaler_y
    
    def subset(self, begin, end):
        subset = TimeSeriesDataset.__new__(TimeSeriesDataset)
        subset.x = self.x[begin:end].clone()
        subset.y = self.y[begin:end].clone()
        subset.time_steps = self.time_steps
        subset.scaler_x = self.scaler_x
        subset.scaler_y = self.scaler_y
        return subset

def load_and_prepare_data(file_path, time_steps, split_date):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    x = df.drop("Federal Funds Rate", axis=1).values
    y = df["Federal Funds Rate"].values
    full_set = TimeSeriesDataset(x, y, time_steps)
    training_set_size = (df.index <= split_date).sum() - time_steps

    return df, full_set.subset(0, training_set_size), full_set.subset(training_set_size, len(full_set))