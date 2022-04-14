import numpy as np
from torch.utils.data import Dataset
import prediction.util as util


class NeuralData(Dataset):
    def __init__(self, csv_path, key_name):
        data = util.fromCSV(csv_path)
        data_x = [[row[key] for row in data] for key in key_name]
        data_y = [row["target"] for row in data]
        self.x = np.array(data_x, dtype="float64")
        self.x = self.x.reshape(self.x.shape[0], len(data_x[0]))
        self.x = self.x.transpose()
        self.y = np.array(data_y, dtype="float64")
        self.y = self.y.transpose()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        res_x = self.x[idx]
        out = self.y[idx]
        return res_x, out
