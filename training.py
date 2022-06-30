import awkward as ak
import numpy as np
from pyrsistent import l
import uproot as uproot
import matplotlib.pyplot as plt
import networkx as nx
from numba import jit
from models import GraphNet, focal_loss
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import mplhep as hep
plt.style.use(hep.style.CMS)

print("++++++ CUDA Device {}".format(torch.cuda.get_device_name()))
import torch_geometric.utils as Utils
# from sklearn.metrics import confusion_matrix
import argparse
from utils import mkdir_p, using, loadData

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--output', type=str,
                    help='model output path')
parser.add_argument('--epochs', type=int, default=20, help="Training epochs")
parser.add_argument('--batch', type=int, default=32, help='Batch size')

args = parser.parse_args()

outputModelPath = args.output + "/"
outputModelCheckpoint = outputModelPath + "checkpoints/"
outputLossFunction = outputModelPath + "loss/"

mkdir_p(outputModelPath)
mkdir_p(outputModelCheckpoint)
mkdir_p(outputLossFunction)

print("===== Loading Files ...")
trainDataset = "./dataProcessed/dataTraining.pt"
valDataset = "./dataProcessed/dataVal.pt"
trainData = torch.load(trainDataset)
valData = torch.load(valDataset)
print("===== Loaded! ")
print("Train data {}".format(trainData))
print("Train data {}".format(trainData))
epochs = args.epochs
batch_size = args.batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GraphNet().to(device)

trainLoader = DataLoader(trainData, batch_size=batch_size)
valLoader = DataLoader(valData, batch_size=batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

train_loss_history = []
val_loss_history = []

print("===== Start training!")
model.train()
for epoch in range(epochs):
    batchloss = []
    for sample in trainLoader:
        optimizer.zero_grad()
        sample.to(device)
        out = model(sample)
        #weight = [1. if l > 0.9 else 1.1 for l in sample.edge_label] # weigh up false edges
        bce_loss = F.binary_cross_entropy(out, sample.edge_label)
        loss = focal_loss(bce_loss, sample.edge_label, 2, 0.4)
        batchloss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    train_loss_history.append(np.mean(batchloss))

    with torch.set_grad_enabled( False ):
        batchloss = []
        for sample in valLoader:
            sample.to(device)
            out = model(sample)
            val_bce = F.binary_cross_entropy(out, sample.edge_label)
            val_loss = focal_loss(val_bce, sample.edge_label, 2, 0.4)
            batchloss.append(val_loss.item())
            
    val_loss_history.append(np.mean(batchloss))
    if(epoch != 0 and epoch % 5 == 0):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_history,
                    }, outputModelCheckpoint+"/epoch_{}.pt".format(epoch))
    print(f"epoch {epoch} - train loss {train_loss_history[-1]} - val loss {val_loss_history[-1]}")

print("==== Saving model {}".format( outputModelPath+"/model.pt"))
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_history,
            }, outputModelPath+"/model.pt")


fig = plt.figure(figsize=(20,15))
plt.plot(train_loss_history, label='train')
plt.plot(val_loss_history, label='val')
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.xticks(range(epochs), range(1, epochs+1))
plt.legend()
plt.savefig(fig, outputLossFunction + "/losses.png")


