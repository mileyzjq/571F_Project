import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tqdm
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import prediction.util as util


class NeuralData(Dataset):
    def __init__(self, csv_path):
        data = util.fromCSV(csv_path)
        data_x = [[row[key] for row in data] for key in ["avgPasses", "isSamePos", "diffInRank", "meanDegree", "betwPerGameP2", "avgPCPercPerP1"]]
        data_y = [row["weight"] for row in data]
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


class NeuralModel(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(NeuralModel, self).__init__()

        # Number of input features is 12.
        self.layer_1 = nn.Linear(n_feature, n_hidden)
        self.layer_2 = nn.Linear(n_hidden, n_hidden)
        self.layer_out = nn.Linear(n_hidden, n_output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        # x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        # x = self.batchnorm2(x)
        # x = self.dropout(x)
        x = self.layer_out(x)

        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train_neural_model(device, model, criterion, optimizer, dataloaders, num_epochs=10):
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    loss_stats = {
        'train': [],
        "val": []
    }

    for e in tqdm.tqdm(range(num_epochs)):
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch.float()).squeeze()
            train_loss = criterion(y_train_pred, y_train_batch.float())
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
        # VALIDATION
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch.float()).squeeze()
                val_loss = criterion(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        print(
            f'Epoch {e + 0:02}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f}')

    return model, loss_stats


def load_dataset(dataset, valid_size):
    train_data = dataset
    test_data = dataset
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    dataset_size = {"train": len(train_idx), "val": len(test_idx)}
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=8)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=8)
    return trainloader, testloader, dataset_size
