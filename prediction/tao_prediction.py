import sys
# append the path
sys.path.append("/Users/michaelma/desktop/workspace/School/ubc/courses/2021-22-Winter-Term2/EECE571F/project/571F_Project")

import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from prediction.trainer import NeuralData, NeuralModel, train_neural_model, load_dataset

parser = argparse.ArgumentParser(description='Soccer')
parser.add_argument('--input_path',
                    default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571F/project/571F_Project/data/processed/player_data2.csv',
                    type=str, help='The input data')
parser.add_argument('--out_path',
                    default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571F/project/571F_Project/trained',
                    type=str, help='Path to save the data')
parser.add_argument('--weight_path',
                    default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571F/project/571F_Project/trained/soccer_1647883884.187955.pth',
                    type=str, help='Path to save the data')
parser.add_argument('--mode', default='train', type=str, help='Select whether to train, evaluate, inference the model')
parser.add_argument('--valid_size', default=0.2, type=float, help='Proportion of data used as validation set')
parser.add_argument('--learning_rate', default=.003, type=float, help='Default learning rate')
parser.add_argument('--epoch', default=10, type=int, help='epoch number')
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
    desired_key_name = ["avg_pass","check_same_postion","check_diff_rank",
        "avg_pass_position","mean_degree","between_P1","avg_pass_percentage_P1"]
    train_data = NeuralData(data_path, desired_key_name)
    train_dataloader, val_dataloader, dataset_size = load_dataset(train_data, args.valid_size)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}
    data_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'val']}

    saved_path = Path(args.out_path)

    saved_path = saved_path.joinpath("{}_{}.pth".format(args.name, time.time()))

    learning_rate = args.learning_rate

    epoch = args.epoch

    model = NeuralModel(7, 128, 1)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model, loss = train_neural_model(device, model, criterion, optimizer, dataloaders, epoch)

    torch.save(best_model.state_dict(), saved_path)

    print("Prediction Model saved to {}".format(saved_path))

elif args.mode.__eq__("eval"):
    print("to eval")

elif args.mode.__eq__("infer"):
    # Load the model
    saved_mode_path = args.weight_path

    model = NeuralModel(6, 128, 1)

    model.load_state_dict(torch.load(saved_mode_path))

    print(model.eval())
