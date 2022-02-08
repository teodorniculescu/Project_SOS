
import os
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math

import csv
def gen_grafic(folder, result, plot):
    x, y, z, x_log = [], [], [], []

    with open(os.path.join(folder, result), "r") as file_object:
        csv_reader = csv.reader(file_object, delimiter=',')
        for elem in csv_reader:
            elem = [float(e) for e in elem]
            learn_rate, momentum, bva = elem
            #lr_aux = math.log(learn_rate, 1e-1)
            lr_aux = 10 - math.log(learn_rate, 1e-1)
            if learn_rate not in x:
                x.append(learn_rate)
                x_log.append(lr_aux)
            if momentum not in y:
                y.append(momentum)
            z.append(bva)

    x, y, z, x_log = np.array(x), np.array(y), np.array(z), np.array(x_log)

    # this changes the x axis to the log values
    x = x_log

    plt.figure(figsize=(8, 8))
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(x, y, z, c=z, cmap='PiYG');
    #z = z.reshape(len(x), len(y))
    #y, x = np.meshgrid(y, x)
    #ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none');

    ax.set_xlabel('learn rate')
    ax.set_ylabel('momentum')
    ax.set_zlabel('accuracy');
    plt.savefig(os.path.join(folder, plot))

    ##############################################

    x, y, z, x_log = [], [], [], []

    with open(os.path.join(folder, result), "r") as file_object:
        csv_reader = csv.reader(file_object, delimiter=',')
        for elem in csv_reader:
            elem = [float(e) for e in elem]
            learn_rate, momentum, bva = elem
            #lr_aux = math.log(learn_rate, 1e-1)
            lr_aux = 10 - math.log(learn_rate, 1e-1)
            if learn_rate not in x:
                x.append(learn_rate)
                x_log.append(lr_aux)
            if momentum not in y:
                y.append(momentum)
            z.append(bva)

    x, y, z, x_log = np.array(x), np.array(y), np.array(z), np.array(x_log)

    # this changes the x axis to the log values
    x = x_log

    for i in range(15):

        plt.figure(figsize=(8, 8))
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter3D(x[i*10:(i+1)*10], y[i*10:(i+1)*10], z[i*10:(i+1)*10], c=z[i*10:(i+1)*10], cmap='PiYG');
        #z = z.reshape(len(x), len(y))
        #y, x = np.meshgrid(y, x)
        #ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none');

        ax.set_xlabel('learn rate')
        ax.set_ylabel('momentum')
        ax.set_zlabel('accuracy');
        plt.savefig(os.path.join(folder, 'gif'+str(i)+".png"))

if __name__ == "__main__":
    folder = 'MODEL_2021_12_07_18_23_00_453717'
    result = 'res.txt'
    plot = 'plot.png'

    gen_grafic(folder, result, plot)
