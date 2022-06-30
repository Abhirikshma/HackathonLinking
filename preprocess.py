import os.path as osp
from utils import loadData
import numpy as np
import pickle
import awkward as ak
import torch
from torch_geometric.data import Dataset, download_url, InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

pickle_data = './dataset_closeByDoublePion/'


print("===== Loading Files ...")
el, e, nf = loadData(pickle_data, num_files = -1)
print("===== Loaded!")

edge_data = []
edge_label = []
node_data = []
print("Preparing dataset ...")
for i,X in enumerate(nf):
    for ev in range(len(X)):
        if len(e[i][ev]) == 0:
            print(f"event {ev} edges {e[i][ev]}")
            continue # skip events with no edges
        else:
            X_ev = []
            edge_data.append(e[i][ev])
            edge_label.append([np.float32(edg) for edg in el[i][ev]])
            for field in X[ev].fields:
                X_ev.append(ak.to_numpy(X[ev][field]))
            node_data.append(X_ev)

data_list = []
for ev in range(len(node_data)):
    x = torch.from_numpy(np.array(node_data[ev]).T)
    e_label = torch.from_numpy(np.array(edge_label[ev]))
    edge_index = torch.from_numpy(edge_data[ev])
    data = Data(x=x, num_nodes=torch.tensor(x.shape[0]), edge_index=edge_index, edge_label=e_label)
    data_list.append(data)

trainRatio = 0.8
valRatio = 0.1
testRatio = 0.1

nSamples = len(data_list)

nTrain = int( trainRatio * nSamples  )
nVal = int( valRatio * nSamples )

trainDataset = data_list[:nTrain]           # training dataset
valDataset = data_list[nTrain:nTrain+nVal]  # validation dataset
testDataset = data_list[nTrain+nVal:]       # test datase

# dataTraining, slicesTraining = InMemoryDataset.collate(trainDataset)
# dataVal, slicesVal = InMemoryDataset.collate(valDataset)
# dataTest, slicesTest = InMemoryDataset.collate(testDataset)
torch.save(trainDataset, './dataProcessed/dataTraining.pt')
torch.save(valDataset, './dataProcessed/dataVal.pt')
torch.save(testDataset, './dataProcessed/dataTest.pt')