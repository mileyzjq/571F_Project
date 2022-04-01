import sys

# append the path
sys.path.append("571F_Project")

import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from prediction.trainer import train_neural_model, load_dataset
from prediction import util
from model.model import NeuralModel, LinearRegression, LogisticRegressionModel
from data.data import NeuralData
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Soccer')
parser.add_argument('--input_path',
                    default='../data/processed/processed_feature.csv',
                    type=str, help='The input data')
parser.add_argument('--out_path',
                    default='../trained',
                    type=str, help='Path to save the data')
parser.add_argument('--weight_path',
                    default='../trained/soccer_1647883884.187955.pth',
                    type=str, help='Path to save the data')
parser.add_argument('--img_path',
                    default='../image',
                    type=str, help='Path to save the data')
parser.add_argument('--mode', default='train', type=str, help='Select whether to train, evaluate, inference the model')
parser.add_argument('--valid_size', default=0.2, type=float, help='Proportion of data used as validation set')
parser.add_argument('--learning_rate', default=.003, type=float, help='Default learning rate')
parser.add_argument('--epoch', default=100, type=int, help='epoch number')
parser.add_argument('--name', default="soccer", type=str, help='Name of the model')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU enable")
else:
    device = torch.device("cpu")
    print("CPU enable")

if args.mode.__eq__("train"):
    data_path = args.input_path
    # desired_key_name = ["avg_pass","check_same_postion","check_diff_rank", "avg_pass_position","mean_degree","between_P1","avg_pass_percentage_P1"]
    # desired_key_name = ["avgPasses","isSamePos","diffInRank","meanDegree","betwPerGameP1","betwPerGameP2",
    #                     "avgPassComplPerP1","avgPassComplPerP2","avgPassAttempPerP1","avgPassAttempPerP2","avgPCPercPerP1","avgPCPercPerP2"]

    key_set1 = ["avgPasses", "isSamePos", "diffInRank", "betwPerGameP1", "betwPerGameP2", "avgPassComplPerP1",
                "avgPassComplPerP2"]
    key_set2 = ["isSamePos", "diffInRank", "betwPerGameP1", "betwPerGameP2", "avgPassComplPerP1", "avgPassComplPerP2"]
    key_set3 = ["avgPasses", "diffInRank", "betwPerGameP1", "betwPerGameP2", "avgPassComplPerP1", "avgPassComplPerP2"]
    key_set4 = ["avgPasses", "isSamePos", "diffInRank", "betwPerGameP1", "betwPerGameP2"]
    key_set5 = ["avgPasses", "isSamePos", "diffInRank", "avgPassComplPerP1", "avgPassComplPerP2"]

    key_set = {"All": key_set1, "Avg Passes": key_set2, "Same Pos.": key_set3,
               "Avg Passes/player": key_set4, "Betweenness": key_set5}

    # key_set_f = {"all": key_set1, "Average Passes": key_set2}

    res_set = {}

    saved_path = Path(args.out_path)

    saved_path = saved_path.joinpath("{}_{}.pth".format(args.name, time.time()))

    img_path = Path(args.img_path)

    train_result_path = img_path.joinpath("train_loss_multiple_{}.png".format(time.time()))

    for key, key_set_feature in key_set.items():
        train_data = NeuralData(data_path, key_set_feature)
        train_dataloader, val_dataloader, dataset_size = load_dataset(train_data, args.valid_size)
        dataloaders = {"train": train_dataloader, "val": val_dataloader}
        data_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'val']}

        learning_rate = args.learning_rate

        epoch = args.epoch

        model = NeuralModel(len(key_set_feature), 256, 1)

        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_model, loss = train_neural_model(device, model, criterion, optimizer, dataloaders, epoch)

        torch.save(best_model.state_dict(), saved_path)

        print("Prediction Model saved to {}".format(saved_path))

        res_set[key] = loss

    for key, loss in res_set.items():
        train_loss = loss["train"]
        val_loss = loss["val"]
        epoch = len(train_loss)
        plt.plot(range(epoch), train_loss, label="{}".format(key))
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.legend(loc='upper right')
    plt.savefig(train_result_path)

    # util.toFig(loss, train_result_path)


elif args.mode.__eq__("eval"):
    print("to eval")

elif args.mode.__eq__("infer"):
    # Load the model
    saved_mode_path = args.weight_path

    model = NeuralModel(6, 128, 1)

    model.load_state_dict(torch.load(saved_mode_path))

    print(model.eval())
