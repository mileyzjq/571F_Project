import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tqdm
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class NeuralData(Dataset):
    def __init__(self, x, y):
        self.x = np.array(x, dtype="float64")
        self.x = self.x.reshape(self.x.shape[0], 2)
        self.y = np.array(y, dtype="float64")

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
        self.layer_out = nn.Linear(n_hidden, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
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

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    for e in tqdm.tqdm(range(num_epochs)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch.float()).squeeze()
            train_loss = criterion(y_train_pred, y_train_batch.float())
            train_acc = binary_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        # VALIDATION
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch.float()).squeeze()
                # y_val_pred = torch.unsqueeze(y_val_pred, 0)
                # y_val_batch = torch.unsqueeze(y_val_batch, 0)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = binary_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))
        print(
            f'Epoch {e + 0:02}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}%| Val Acc: {val_epoch_acc / len(val_loader):.3f}%')

    return model, loss_stats, accuracy_stats


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
                                              sampler=train_sampler, batch_size=1)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=1)
    return trainloader, testloader, dataset_size
