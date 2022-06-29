import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
# import tensorflow as tf
import glob
from numba import jit
import pickle
import os, errno

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

@jit
def computeEdgeAndLabels(trk_data, ass_data, gra_data, nodes, edges, edges_labels):
    '''Compute the truth graph'''
    for i in range(trk_data.NTracksters):
        nodes.append(i)
        qualities = ass_data.tsCLUE3D_recoToSim_CP_score[i]
        best_sts_i = ass_data.tsCLUE3D_recoToSim_CP[i][ak.argmin(qualities)]
        best_sts_i = best_sts_i if qualities[best_sts_i]<0.1 else -1
        for j in gra_data.linked_inners[i]:
            edges.append([j,i])
            qualities = ass_data.tsCLUE3D_recoToSim_CP_score[j]
            best_sts_j = ass_data.tsCLUE3D_recoToSim_CP[j][ak.argmin(qualities)]
            best_sts_j = best_sts_j if qualities[best_sts_j]<0.1 else -1
            if best_sts_i == best_sts_j:
                edges_labels.append(1)
            else:
                edges_labels.append(0)
                
input_folder = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/hackathon/samples/close_by_double_pion/production/new_new_ntuples/"
files = glob.glob(f"{input_folder}/*ntuples_*.root")

calos = [ ]
tracksters = [ ]
associations = [ ]
graph = [ ]

X = [ ]
Edges = [ ]
Edges_labels = [ ] 
outputPath  = './dataset_closeByDoublePion/'
mkdir_p(outputPath)

N = 10000000
offset = 13
for i_file, file in enumerate(files[offset:]):
    i_file += offset
    if i_file >= N: break
    try:
        f = uproot.open(file)
        t =  f["ntuplizer/tracksters"]
        calo = f["ntuplizer/simtrackstersCP"]
        ass = f["ntuplizer/associations"]
        gra = f["ntuplizer/graph"]
        
        trk_data = t.arrays(["NTracksters", "raw_energy","raw_em_energy","barycenter_x","barycenter_y","barycenter_z","eVector0_x", "eVector0_y","eVector0_z"])
        gra_data = gra.arrays(['linked_inners'])
        ass_data = ass.arrays([ "tsCLUE3D_recoToSim_CP", "tsCLUE3D_recoToSim_CP_score"])

        X = [ ]
        Edges = [ ]
        Edges_labels = [ ] 
    
    except:
        print("error ", file)
        continue
    print('\nProcessing file {} '.format(file), end="")
    
    for ev in range(len(gra_data)):
        print(".", end="")


        # Save the input variables

        x_ev = ak.zip({"raw_en": trk_data[ev].raw_energy, 
                       'raw_em_energy': trk_data[ev].raw_em_energy,
                         "barycenter_x": trk_data[ev].barycenter_x,
                         "barycenter_y": trk_data[ev].barycenter_y,
                         "barycenter_z": trk_data[ev].barycenter_z,
                         "eVector0_x": trk_data[ev].eVector0_x,
                         "eVector0_y": trk_data[ev].eVector0_y,
                         "eVector0_z": trk_data[ev].eVector0_z})

        X.append(x_ev)
        nodes = []
        edges = []
        edges_labels = []
        
        computeEdgeAndLabels(trk_data[ev], ass_data[ev], gra_data[ev], nodes, edges, edges_labels)
        ed_np = np.array(edges).T
        Edges.append(ed_np)
        Edges_labels.append(edges_labels)
            
        # Save to disk
        if((ev % 200 == 0 and ev != 0)  or (ev == len(gra_data))):
            print("Saving now the pickle data {} {}".format(i_file,str(ev)))

            pickle_dir = outputPath
            with open(pickle_dir+"{}_{}_node_features.pkl".format(str(i_file), str(ev)), "wb") as fp:   #Pickling
                pickle.dump(X, fp)
            with open(pickle_dir+"{}_{}_edges.pkl".format(str(i_file),str(ev)), "wb") as fp:   #Pickling
                pickle.dump(Edges, fp)
            with open(pickle_dir+"{}_{}_edges_labels.pkl".format(str(i_file),str(ev)), "wb") as fp:   #Pickling
                pickle.dump(Edges_labels, fp)
            #Emptying arrays
            ed_np = []
            Edges = []
            Edges_labels = []    
            X = []
        
