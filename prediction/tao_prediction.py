import argparse
import time
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
import numpy as np

import torch
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from prediction.trainer import NeuralData, NeuralModel, train_neural_model, load_dataset


parser = argparse.ArgumentParser(description='Soccer')
parser.add_argument('--mode', default='train', type=str, help='Select whether to train, evaluate, inference the model')
parser.add_argument('--valid_size', default=0.5, type=float, help='Proportion of data used as validation set')
parser.add_argument('--learning_rate', default=.003, type=float, help='Default learning rate')
parser.add_argument('--epoch', default=10, type=int, help='epoch number')
parser.add_argument('--name', default="best_attack", type=str, help='Name of the model')
args = parser.parse_args()

train_x1 = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]]
train_y1 = [0, 1, 1, 0, 0, 1, 1, 0]
train_x2 = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
train_y2 = [-1, 1, 1, -1]

train_data = NeuralData(train_x1, train_y1)
train_dataloader, val_dataloader, dataset_size = load_dataset(train_data, args.valid_size)
dataloaders = {"train": train_dataloader, "val": val_dataloader}
data_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'val']}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU enable")
else:
    device = torch.device("cpu")
    print("CPU enable")

if args.mode.__eq__("train"):
    learning_rate = args.learning_rate

    epoch = args.epoch

    model = NeuralModel(2, 64, 1)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model, loss, acc = train_neural_model(device, model, criterion, optimizer, dataloaders, epoch)

elif args.mode.__eq__("eval"):
    print("to eval")
elif args.mode.__eq__("infer"):
    print("to infer")
