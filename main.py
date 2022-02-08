import torchvision
import random
import os
import torch
from torch import nn
from datetime import datetime
import time
import argparse
import torch.optim as optim
from tqdm import tqdm
import csv
import cv2
import numpy as np
from time import time

from plot import gen_grafic

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx   

def parser():
    parser = argparse.ArgumentParser(description="SOS")

    parser.add_argument('--learn_rate', nargs='+', type=float, required=True)
    parser.add_argument('--momentum', nargs='+', type=float, required=True)


    parser.add_argument("--batch_size", default=256, type=int,)
    parser.add_argument("--epochs", default=10, type=int,)

    parser.add_argument("--device_name", default='cuda:0', type=str,)

    parser.add_argument("--channels", default=1, type=int,)
    parser.add_argument("--width", default=28, type=int,)
    parser.add_argument("--height", default=28, type=int,)
    parser.add_argument("--num_output_classes", default=10, type=int,)


    parser.add_argument('--shuffle', default='True', choices=['True', 'False'])
    parser.add_argument('--num_workers', default=4, type=int,)

    parser.add_argument('--gridsearch_type', required=True, choices=['all', 'pso'])

    args = parser.parse_args()

    args.shuffle = args.shuffle == 'True'

    return args


def check_arguments_errors(args):
    if len(args.learn_rate) != 2:
        raise Exception("Wrong interval for learn rate", args.learn_rate)
    if len(args.momentum) != 2:
        raise Exception("Wrong interval for momentum", args.momentum)


class CustomModel(nn.Module):
    def __init__(self, args, model_type):
        super(CustomModel, self).__init__()     
        self.args = args
        if model_type == 1:
            self.type1()
        elif model_type == 2:
            self.type2()
        else:
            raise Exception("Unknown type")

    def type1(self):
        self.net = []
        in_channels = self.args.channels

        for out_channels in [16, 32]:
            self.net += [
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ]
            in_channels = out_channels

        self.net.append(nn.Flatten())

        random_t = torch.rand(2, self.args.channels, self.args.height, self.args.width)
        random_t = nn.Sequential(*self.net)(random_t)
        self.net.append(nn.Linear(random_t.shape[1], self.args.num_output_classes))
        self.net.append( nn.Softmax(dim=1))
        self.net = nn.Sequential(*self.net)


    def type2(self):
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.args.height*self.args.width*self.args.channels, 40),
            nn.Linear(40, self.args.num_output_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        return self.net(x)

def train(
        p,
        args, 
        traindataset,
        validdataset,
        trainloader,
        validloader,
        use_print=False,
    ):
    print(p)
    learn_rate, momentum = p


    device = args.device_name
    net = CustomModel(args, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    parameters = net.parameters()

    optimizer = optim.SGD(parameters, lr=learn_rate, momentum=momentum)

    json_data = {}
    best_valid_acc = None
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        start = time()

        net.train()
        train_loss = 0.0
        train_acc = 0.0
        for inputs, labels in tqdm(trainloader, disable=not use_print):
            inputs, labels = inputs.to(device), labels.to(device)
            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(preds == labels.data)

        train_loss /= len(traindataset) 
        train_acc = train_acc.double() / len(traindataset)
        train_acc = train_acc.item()

        torch.cuda.empty_cache()

        if use_print:
            print('Val Stage')
        net.eval()     # Optional when not using Model Specific layer
        valid_loss = 0.0
        valid_acc = 0.0
        for inputs, labels in tqdm(validloader, disable=not use_print):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            valid_loss += loss.item() * inputs.size(0)
            valid_acc += torch.sum(preds == labels.data)

        valid_loss /= len(validdataset) 
        valid_acc = valid_acc.double() / len(validdataset)
        valid_acc = valid_acc.item()

        torch.cuda.empty_cache()


        end = time()

        if use_print:
            print(str(epoch) + '/' + str(args.epochs) + '-> ' + ' train_acc: ' + str(train_acc) + ' | valid_acc: ' + str(valid_acc) + ' train_loss: ' + str(train_loss) + ' | valid_loss: ' + str(valid_loss) + ' | epoch sec: ' + str(end-start) + ' | time left: ' + str((args.epochs - epoch - 1) * (end-start) / 60) + 'min')

        json_data[epoch] = {}
        json_data[epoch]['train_loss'] = train_loss
        json_data[epoch]['valid_loss'] = valid_loss 
        json_data[epoch]['train_acc'] = train_acc
        json_data[epoch]['valid_acc'] = valid_acc 

        if best_valid_acc is None or best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
    
    return best_valid_acc


def pso_train(
        plist,
        args, 
        traindataset,
        validdataset,
        trainloader,
        validloader,
        use_print=False,
    ):
    res = []
    for p in plist:
        p = (1e-1**(10 - p[0]), p[1])

        bva = train(
            p,
            args, 
            traindataset,
            validdataset,
            trainloader,
            validloader,
            use_print=False,
        )

        with open(os.path.join(args.folder, args.result), "a") as file_object:
            csv_writer = csv.writer(file_object, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow((p[0], p[1], bva))

        res.append(1-bva)

    return np.array(res)

class CubicFunction:
    def evaluate(self, x):
        return x ** 2 + torch.exp(x)

def gridsearch_all(args, traindataset, validdataset, trainloader, validloader,):
    all_gridsearch_list = []
    for learn_rate in np.arange(1, 9 + 0.5, 0.5):
        learn_rate = 1e-1 ** learn_rate
        for momentum in np.arange(0, 1 + 0.05, 0.05):
            all_gridsearch_list.append([learn_rate, momentum])

    json_data = []
    for learn_rate, momentum in tqdm(all_gridsearch_list):
        #bva = random.uniform(1.5, 1.9)
        bva = train(
                (learn_rate, momentum),
                args,
                traindataset,
                validdataset,
                trainloader,
                validloader,
            )

        json_data.append([learn_rate, momentum, bva])

        with open(os.path.join(args.folder, args.result), "w") as file_object:
            csv_writer = csv.writer(file_object, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for elem in json_data:
                csv_writer.writerow(elem)

def gridsearch_pso(args, traindataset, validdataset, trainloader, validloader,):
    options = {'c1': 0.1, 'c2': 0.1, 'w': 0.8}
    bounds = ([1, 0], [9, 1])
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

    cost, pos = optimizer.optimize(pso_train, iters=15, args=args, traindataset=traindataset,
            validdataset=validdataset, trainloader=trainloader, validloader=validloader)
    print(cost, pos)

def config_args(args):
    args.folder = 'MODEL_' + f"{datetime.now()}"
    args.folder = args.folder.replace(' ', '_')
    args.folder = args.folder.replace(':', '_')
    args.folder = args.folder.replace('.', '_')
    args.folder = args.folder.replace('-', '_')

    args.result = 'res.txt'
    args.plot = 'plot.png'

    os.mkdir(args.folder)

    return args

def gen_dataset(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    traindataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    validdataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    return traindataset, validdataset, trainloader, validloader 

def main():
    args = parser()
    check_arguments_errors(args)
    args = config_args(args)

    traindataset, validdataset, trainloader, validloader = gen_dataset(args)


    start = time()
    if args.gridsearch_type == 'all':
        gridsearch_all(args, traindataset, validdataset, trainloader, validloader,)
    elif args.gridsearch_type == 'pso':
        gridsearch_pso(args, traindataset, validdataset, trainloader, validloader,)
    else:
        raise Exception("Unknown gridsearch type", args.gridsearch_type)
    end = time()
    print('Gridsearch elapsed time', end - start)
    
    gen_grafic(args.folder, args.result, args.plot)

if __name__ == "__main__":
    main()
