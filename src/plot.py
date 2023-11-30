# -*- coding: utf-8 -*-
"""
Created on Thu 24 15:45 2022

visualization

@author: NemotoS
"""

import matplotlib.pyplot as plt
import numpy as np
        
def plot_loss(num_epochs,train_loss_list,valid_loss_list=[],dir_name=""):
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(111)
    plt.rcParams["font.size"] = 18
    ax1.plot(train_loss_list,color="blue",label="train")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.grid()

    if len(valid_loss_list) > 0:
        n = len(train_loss_list) // len(valid_loss_list)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(n,len(train_loss_list)+1,n),valid_loss_list,color="orange",label="valid")
        ax2.grid()

    if len(dir_name) > 0:
        plt.savefig(dir_name+"/loss.png",bbox_inches="tight")
    else:
        plt.show()

def plot_accuracy(num_epochs,accuracy_list,dir_name=""):
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(111)
    plt.rcParams["font.size"] = 18
    ax1.plot(accuracy_list,color="blue",label="train")
    ax1.set_xlabel("step")
    ax1.set_ylabel("accuracy")
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.grid()


    if len(dir_name) > 0:
        plt.savefig(dir_name+"/accuracy.png",bbox_inches="tight")
    else:
        plt.show()