import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import prediction.util as util

class LinearRegressionData(Dataset):
    def __init__(self, csv_path, key_name):
        data = util.fromCSV(csv_path)
        data_x = [[row[key] for row in data] for key in key_name]
        data_y = [row["target"] for row in data]
        self.x = np.array(data_x, dtype=np.float32)
        self.x = self.x.reshape(self.x.shape[0], len(data_x[0]))
        self.x = self.x.transpose()
        self.y = np.array(data_y, dtype=np.float32)
        self.y = self.y.transpose()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        res_x = self.x[idx]
        out = self.y[idx]
        return res_x, out


class LinearRegression(nn.Module):
    def __init__(self, n_feature, n_output):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(n_feature, n_output)

    def forward(self, x):
        return self.linear(x)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
