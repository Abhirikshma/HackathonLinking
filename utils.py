import resource
import glob
import pickle
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

#Create new directory
def mkdir_p(mypath):
    '''Function to create a new directory, if it not already exist
        - mypath : directory path
    '''
    from errno import EEXIST
    from os import makedirs,path
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                usage[2]/1024.0 )

def loadData(path, num_files = -1, lc_data = False):
    f_edges_label = glob.glob(path + "*edges_labels.pkl" )
    f_edges = glob.glob(path + "*edges.pkl" )
    f_nodes_features = glob.glob(path + "*node_features.pkl" )
    if lc_data:
        f_energy_layer = glob.glob(path + "*energies_on_layers.pkl")
    en_lay = []
    edges_label = []
    edges = []
    nodes_features = []
    

    for i_f, _ in enumerate(f_edges_label):
        if(num_files == -1):
            n = len(f_edges_label)
        else:
            n = num_files
        if(i_f <= n):
            f = f_edges_label[i_f]
            with open(f, 'rb') as fb:
                edges_label.append(pickle.load(fb))
            f = f_edges[i_f]
            with open(f, 'rb') as fb:
                edges.append(pickle.load(fb))
            f = f_nodes_features[i_f]
            with open(f, 'rb') as fb:
                nodes_features.append(pickle.load(fb))
            if lc_data:
                f = f_energy_layer[i_f]
                try:
                    with open(f, 'rb') as fb:
                        en_lay.append(pickle.load(fb))
                except:
                    print(f"error in energy_on_layer {i_f}")
        else:
            break
    return edges_label, edges, nodes_features, en_lay

# test dataset is not normalized before storing on disk
# as the mean_ and scale_ are required for the 
# inverse transformation
def normalize_and_get_data(data_list, ev):
    data_list_ev = data_list[ev]
    x_np = data_list_ev[0]
    x_coord_slice = x_np[:, [0,1,2]] # barycenter x, y, z
    x_rest_slice = x_np[:, [9,10,11]] # size, raw_e, raw_em_e
    x_en_lay_digest = data_list_ev[-1]
    
    mean = []
    std = []
    scaler=StandardScaler()
    scaler.fit(x_coord_slice)
    x_coord_norm = scaler.transform(x_coord_slice)
    mean.append(scaler.mean_)
    std.append(scaler.scale_)
    
    mean.append(np.zeros(6)) # for the unnormalized features
    std.append(np.ones(6))
    
    scaler.fit(x_rest_slice)
    x_rest_norm = scaler.transform(x_rest_slice)
    mean.append(scaler.mean_)
    std.append(scaler.scale_)
    
    mean = np.concatenate(mean, axis=-1)
    std = np.concatenate(std, axis=-1)
    
    x_ev = torch.from_numpy(np.concatenate((x_coord_norm, x_np[:,[3,4,5,6,7,8]], x_rest_norm), axis=1))
    edge_label = torch.from_numpy(data_list_ev[1])
    edge_index = torch.from_numpy(data_list_ev[2])
    data = Data(x=x_ev, num_nodes=torch.tensor(x_ev.shape[0]), edge_index=edge_index, edge_label=edge_label, en_hgcal_layers = torch.from_numpy(np.array(x_en_lay_digest)))
    
    return data, mean, std

# list of indices of best matched simts to all ts in an event
def get_truth_labels(data_list, ev):
    data_list_ev = data_list[ev]
    x_np = data_list_ev[0]
    x_best_simts = x_np[:, 12]
    return x_best_simts

# candidates containing the trackster
def get_cand_labels(data_list, ev):
    data_list_ev = data_list[ev]
    return data_list_ev[0][:, 13]