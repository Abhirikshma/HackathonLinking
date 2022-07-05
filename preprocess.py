import os.path as osp
from utils import loadData
import numpy as np
from sklearn.preprocessing import StandardScaler
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
                if "sigma" in field:
                    continue
                X_ev.append(np.float32(ak.to_numpy(X[ev][field])))
            node_data.append(X_ev)

data_list = []
print(f"{len(node_data)} total events in dataset")

trainRatio = 0.8
valRatio = 0.1
testRatio = 0.1

nSamples = len(node_data)

nTrain = int( trainRatio * nSamples  )
nVal = int( valRatio * nSamples )

for ev in range(len(node_data[:nTrain+nVal])):
    x_np = np.array(node_data[ev]).T
    x_coord_slice = x_np[:, [0,1,2]]
    x_rest_slice = x_np[:, [9,10,11]]
    
    scaler = StandardScaler()
    scaler.fit(x_coord_slice)
    x_coord_norm = scaler.transform(x_coord_slice)

    scaler.fit(x_rest_slice)
    x_rest_norm = scaler.transform(x_rest_slice)
     
    x_norm = np.concatenate((x_coord_norm, x_np[:,[3,4,5,6,7,8]], x_rest_norm), axis=1)
    
    x = torch.from_numpy(x_norm)
    e_label = torch.from_numpy(np.array(edge_label[ev]))
    edge_index = torch.from_numpy(edge_data[ev])
    data = Data(x=x, num_nodes=torch.tensor(x.shape[0]), edge_index=edge_index, edge_label=e_label)
    data_list.append(data)
    
# test split is not normalized here
test_data_list = []
for ev in range(len(node_data[nTrain+nVal:])):
    x_np = np.array(node_data[ev]).T
    '''
    x_coord_slice = x_np[:, [0,1,2]]
    x_rest_slice = x_np[:, [9,10,11,12,13,14]]
    
    scaler = StandardScaler()
    scaler.fit(x_coord_slice)
    x_coord_norm = scaler.transform(x_coord_slice)

    scaler.fit(x_rest_slice)
    x_rest_norm = scaler.transform(x_rest_slice)
     
    x_norm = np.concatenate((x_coord_norm, x_np[:,[3,4,5,6,7,8]], x_rest_norm), axis=1)
    '''
    
    #x = torch.from_numpy(x_np)
    #e_label = torch.from_numpy(np.array(edge_label[ev]))
    #edge_index = torch.from_numpy(edge_data[ev])
    #data = Data(x=x, num_nodes=torch.tensor(x.shape[0]), edge_index=edge_index, edge_label=e_label)
    data = [x_np, np.array(edge_label[ev]), edge_data[ev]]
    test_data_list.append(data)



trainDataset = data_list[:nTrain]           # training dataset
valDataset = data_list[nTrain:]  # validation dataset
testDataset = test_data_list       # test datase

# dataTraining, slicesTraining = InMemoryDataset.collate(trainDataset)
# dataVal, slicesVal = InMemoryDataset.collate(valDataset)
# dataTest, slicesTest = InMemoryDataset.collate(testDataset)
torch.save(trainDataset, './dataProcessed/dataTraining.pt')
torch.save(valDataset, './dataProcessed/dataVal.pt')
torch.save(testDataset, './dataProcessed/dataTest.pt')
print("===== Saved dataset!")