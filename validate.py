import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.utils as Utils
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from scipy import linalg
import networkx as nx
import mplhep as hep
from numba import jit
import matplotlib.colors
from pylab import cm
#from models import GraphNet
plt.style.use(hep.style.CMS)
from utils import normalize_and_get_data
from utils import get_truth_labels
from utils import get_cand_labels

class GraphNet_noLC(nn.Module):
    def __init__(self, input_dim = 12, hidden_dim = 64, output_dim = 1, aggr = 'add', niters = 4):
        super(GraphNet_noLC, self).__init__()
        
        # transform to latent space
        self.inputnetwork = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # to compute messages
        convnetwork = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # EdgeConv
        self.graphconv = EdgeConv(nn=convnetwork, aggr=aggr)
        
        # edge features from node embeddings for classification
        self.edgenetwork = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.niters = niters
    
    def forward(self, data):
        X = data.x
        H = self.inputnetwork(X)
        for i in range(self.niters):
            (prepared_edges, _) = Utils.add_self_loops(data.edge_index)
            H = self.graphconv(H, Utils.to_undirected(prepared_edges))
            
        src, dst = data.edge_index
        return self.edgenetwork(torch.cat([H[src], H[dst]], dim=-1)).squeeze(-1)

class GraphNet(nn.Module):
    def __init__(self, input_dim = 14, hidden_dim = 64, output_dim = 1, aggr = 'add', niters = 4):
        super(GraphNet, self).__init__()
        
        # transform to latent space
        self.inputnetwork = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # to compute messages
        convnetwork = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # EdgeConv
        self.graphconv = EdgeConv(nn=convnetwork, aggr=aggr)
        
        # edge features from node embeddings for classification
        self.edgenetwork = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.niters = niters
    
    def forward(self, data, device = "cuda"):
        X = data.x
        en_lay = data.en_hgcal_layers
        en_lay = en_lay.cpu()
        en_lay_in = torch.from_numpy(np.float32(en_lay))
        
        # if(device == 'cuda'):
        #     en_lay_in = en_lay_in.cuda()
        
        features = en_lay_in
        X = torch.cat([X, features], dim = -1)
        H = self.inputnetwork(X)
        for i in range(self.niters):
            (prepared_edges, _) = Utils.add_self_loops(data.edge_index)
            H = self.graphconv(H, Utils.to_undirected(prepared_edges))
            
        src, dst = data.edge_index
        return self.edgenetwork(torch.cat([H[src], H[dst]], dim=-1)).squeeze(-1)

    

def truth_pairs(model, data_list, ev, thr=0.5):
    data_ev, mean, std = normalize_and_get_data(data_list, ev)
    truth_edge_index = data_ev.edge_index
    truth_edge_label = data_ev.edge_label > thr
    truth_nodes_features = torch.add(torch.mul(data_ev.x, torch.from_numpy(std)),torch.from_numpy(mean))
    
    src_edge_index_true = truth_edge_index[0][truth_edge_label]
    dest_edge_index_true = truth_edge_index[1][truth_edge_label]

    index_tuple = []
    for i in range(len(src_edge_index_true)):
        index_tuple.append([src_edge_index_true[i], dest_edge_index_true[i]])
    return truth_nodes_features, index_tuple


def connectivity_matrix(model, data_list, ev, similarity=True, thr=0.6):
    data_ev, mean, std = normalize_and_get_data(data_list, ev)
    out = model(data_ev)
    N = data_ev.num_nodes
    mat = np.zeros([N, N])
    truth_mat = np.zeros([N, N])
    for indx, src in enumerate(data_ev.edge_index[0]):
        dest = data_ev.edge_index[1][indx]
        # weighted adj is filled only if score > thr
        if out[indx] > thr:
            mat[src][dest] = out[indx]
            mat[dest][src] = out[indx]
            
        truth_mat[src][dest] = data_ev.edge_label[indx]
        truth_mat[dest][src] = data_ev.edge_label[indx]
        
    if similarity == False:
        mat = mat > thr
    return mat, truth_mat

#@jit
def scores_reco_to_sim(predicted_clusters, truth_cluster_labels, truth_cluster_energies, E):
    num_truth_clusters = int(max(truth_cluster_labels)+1)
    pred_cluster_energies = []
    reco_sim_scores = []
    best_sim_matches = []
    
    for pred_cluster in predicted_clusters:
        # for each predicted cluster, one score for every truth cluster
        scores_cluster = np.zeros(num_truth_clusters) # [0 for i in range(num_truth_clusters)]
        clusterE = 0
        for trackster in pred_cluster:
            truth_label = int(truth_cluster_labels[trackster])
            scores_cluster[truth_label] += E[trackster]
            clusterE += E[trackster]
        pred_cluster_energies.append(clusterE)
        
        for cluster in range(num_truth_clusters):
            scores_cluster[cluster] /= (pred_cluster_energies[-1] + truth_cluster_energies[cluster] - scores_cluster[cluster])
        reco_sim_scores.append(np.max(scores_cluster))
        best_sim_matches.append(np.argmax(scores_cluster))
        
    return reco_sim_scores, best_sim_matches, pred_cluster_energies

# Load the dataset
testDataset = torch.load("/eos/user/a/abhiriks/SWAN_projects/TICLv4graph/test/HackathonLinking/dataProcessed_closeByDoublePion_LC/dataTest.pt")

# Load the model
modelLoad = GraphNet()
modelLoad.load_state_dict(torch.load("/eos/user/a/abhiriks/SWAN_projects/TICLv4graph/test/HackathonLinking/model/trackster_graphconv_LC.pt", map_location=torch.device("cpu")))
modelLoad.eval()

# model w/o LC
model_noLC = GraphNet_noLC()
model_noLC.load_state_dict(torch.load("/eos/user/a/abhiriks/SWAN_projects/TICLv4graph/test/HackathonLinking/model/trackster_graphconv_improved_truth.pt", map_location=torch.device("cpu")))
model_noLC.eval()

scores = []
scores_noLC = []
scores_cand = []
pred_energies = []
pred_energies_noLC = []
candidate_energies  = []

more_truth_clusters = 0
isolated_truth_cluster = 0
events_processed = 0
tot_truth_clusters = 0
tot_clue3d_tracksters = 0

for ev in range(len(testDataset[:10000])):
    if ev%100 == 0:
        print(f"event {ev}")

    # get the truth and predicted (weighted) adj matrices for the event
    adj_weighted, truth_adj = connectivity_matrix(modelLoad, testDataset, ev)
    adj_weighted_noLC, truth_adj_noLC = connectivity_matrix(model_noLC, testDataset, ev)

    degree_weighted = np.sum(adj_weighted, axis = 1)
    degree_weighted_noLC = np.sum(adj_weighted_noLC, axis = 1)

    D_weigh = np.diag(degree_weighted)
    D_weigh_noLC = np.diag(degree_weighted_noLC)

    L = D_weigh - adj_weighted # Laplacian
    L_noLC = D_weigh_noLC -adj_weighted_noLC
    
    if np.shape(L)[0] < 2 or np.shape(L_noLC)[0] < 2:
        print("too few tracksters, continuing ...")
        continue

    # compute eigenvalues/vectors of the Laplacian
    n_eigvals = min(np.shape(L)[0]-2, 9)
    n_eigvals_noLC = min(np.shape(L_noLC)[0]-2, 9)
    try:
        eigvals_sorted, eigvecs_sorted = linalg.eigh(L, D_weigh, subset_by_index = [0, n_eigvals])
    except linalg.LinAlgError:
        eigvals_sorted, eigvecs_sorted = linalg.eigh(L, subset_by_index = [0, n_eigvals])

    try:
        eigvals_sorted_noLC, eigvecs_sorted_noLC = linalg.eigh(L_noLC, D_weigh_noLC, subset_by_index = [0, n_eigvals_noLC])
    except linalg.LinAlgError:
        eigvals_sorted_noLC, eigvecs_sorted_noLC = linalg.eigh(L_noLC, subset_by_index = [0, n_eigvals_noLC]) 

    # get the number of clusters from the eigenvalue spectrum
    knee = 0
    eigenvals_diff = np.diff(eigvals_sorted)
    for i, d in enumerate(eigenvals_diff):
        if d > 0.2:
            knee = i
            if i == len(eigenvals_diff)-1:
                break
            if eigenvals_diff[i+1] > 3*d:
                knee = knee + 1
            break

    n_clusters = knee + 1

    knee = 0
    eigenvals_diff = np.diff(eigvals_sorted_noLC)
    for i, d in enumerate(eigenvals_diff):
        if d > 0.2:
            knee = i
            if i == len(eigenvals_diff)-1:
                break
            if eigenvals_diff[i+1] > 3*d:
                knee = knee + 1
            break

    n_clusters_noLC = knee + 1

    # Spectral clustering
    
    sc = SpectralClustering(n_clusters = n_clusters, affinity = "precomputed", assign_labels="cluster_qr")
    pred_cluster_labels = sc.fit_predict(adj_weighted)

    num_pred_clusters = max(pred_cluster_labels)+1
    predicted_clusters = [[] for i in range(num_pred_clusters)]

    for i, label in enumerate(pred_cluster_labels):
        predicted_clusters[label].append(i)

    sc = SpectralClustering(n_clusters = n_clusters_noLC, affinity = "precomputed", assign_labels="cluster_qr")
    pred_cluster_labels_noLC = sc.fit_predict(adj_weighted_noLC)

    num_pred_clusters_noLC = max(pred_cluster_labels_noLC)+1
    predicted_clusters_noLC = [[] for i in range(num_pred_clusters_noLC)]

    for i, label in enumerate(pred_cluster_labels_noLC):
        predicted_clusters_noLC[label].append(i)

    # Calculate truth clustering 
    
    truth_cluster_labels = get_truth_labels(testDataset, ev)
    num_truth_clusters = int(max(truth_cluster_labels)+1)
    
    tot_truth_clusters += num_truth_clusters
    tot_clue3d_tracksters += len(truth_cluster_labels)

    truth_clusters = [[] for i in range(num_truth_clusters)]

    for t, l in enumerate(truth_cluster_labels):
        truth_clusters[int(l)].append(t)
        
    # # "clustering" from TICL Candidates
    
    # cluster_labels_candidate = get_cand_labels(testDataset, ev)
    # clusters_candidate = [[] for i in range(int(max(cluster_labels_candidate))+1)]
    # for ts, cand in enumerate(cluster_labels_candidate):
    #     clusters_candidate[int(cand)].append(ts)
    
    # (hacked from the edge labels; this has the problem 
    # that far away tracksters which do not have any edges to other 
    # tracksters in the input graph are treated as separate "clusters" this way.
    # should ideally be the best simSTS match from the associations)
    
    '''
    t_node, t_pairs = truth_pairs(modelLoad, testDataset, ev, 0.5)

    t_edges = []
    for p in t_pairs:
        t_edges.append([p[0].item(), p[1].item()])
    G = nx.Graph()
    G.add_edges_from(t_edges)
    G.add_nodes_from(range(len(t_node)))
    num_truth_clusters = nx.number_connected_components(G)
    if num_truth_clusters > 2:
        more_truth_clusters += 1
    
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    lone_cluster = False
    truth_cluster_labels = np.zeros(len(t_node))
    truth_clusters = []
    for i, s in enumerate(S):
        this_cluster = []
        for n in list(s):
            truth_cluster_labels[n] = i
            this_cluster.append(n)
        truth_clusters.append(this_cluster)
        if len(this_cluster) == 1:
            lone_cluster = True

    if num_truth_clusters > 2 and lone_cluster:
        isolated_truth_cluster += 1
    '''

    # Validation score between every "super trackster" and simtrackster
    # a trackster is "in" it's best matched simtrackster
    # Calculated as intersection / union; 1 = perfect match
    # Intersection = sum(Energy of tracksters common)
    # for every "super trackster" consider only the best score

    E = testDataset[ev][0][:, 10]
    truth_cluster_energies = []
    for t_cluster in truth_clusters:
        clusterE = 0
        for t in t_cluster:
            clusterE += E[t]
        truth_cluster_energies.append(clusterE)

    # Super Trackster to sim TS (from CP) scores
    reco_sim_scores, best_sim_matches, pred_cluster_energies = scores_reco_to_sim(predicted_clusters, truth_cluster_labels, truth_cluster_energies, E)

    # scores for Super Tracksters without LC
    reco_sim_scores_noLC, best_sim_matches_noLC, pred_cluster_energies_noLC = scores_reco_to_sim(predicted_clusters_noLC, truth_cluster_labels, truth_cluster_energies, E)
    
    # TICL Candidate to sim TS (from CP) scores
    # cand_sim_scores, cand_best_sim_matches, cand_energies = scores_reco_to_sim(clusters_candidate, truth_cluster_labels, truth_cluster_energies, E)

    for s in reco_sim_scores:
        scores.append(s)
    for s in reco_sim_scores_noLC:
        scores_noLC.append(s)
    # for s in cand_sim_scores:
    #     scores_cand.append(s)
        
    for en in pred_cluster_energies:
        pred_energies.append(en)
    for en in pred_cluster_energies_noLC:
        pred_energies_noLC.append(en)
    # for en in cand_energies:
    #     candidate_energies.append(en)
        
    events_processed += 1
    
    """ print(f"RECO-SIM scores : {reco_sim_scores}")
    print(f"Best SIM matches : {best_sim_matches}")
    print(f"\nPredicted cluster energies (GeV) : {pred_cluster_energies}")
    print(f"Truth cluster energies (GeV) : {truth_cluster_energies}") """

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
# ax.hist(scores_cand, alpha=0.6, bins=20, label="TICL Candidate")
ax.hist(scores, alpha=0.6, bins=20, label="GNN SuperTrackster")
ax.hist(scores_noLC, alpha=0.6, bins=20, label="GNN SuperTrackster noLC")
ax.set_xlabel(r"Score (Energy $\cap$ over $\cup$)", fontsize=16)
ax.set_ylabel(r"Entries", fontsize=16)
#ax.set_title(r"Reco $\it{Super}$Trackster to SimTrackster from CP", fontsize=16)
plt.legend()
# plt.savefig("scores_histogram_LC_linear.png")
plt.yscale("log")
# plt.savefig("scores_histogram_LC.png")

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
ax.scatter(scores, pred_energies, marker=".")
ax.set_xlabel(r"Score (Energy $\cap$ over $\cup$)", fontsize=16)
ax.set_ylabel(r"Energy of $\it{Super}$Trackster [GeV]", fontsize=16)
ax.set_title("GNN SuperTrackster", fontsize=18)
# plt.savefig("EnergyVsScore_LC.png")

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
ax.scatter(scores_noLC, pred_energies_noLC, marker=".")
ax.set_xlabel(r"Score (Energy $\cap$ over $\cup$)", fontsize=16)
ax.set_ylabel(r"Energy of $\it{Super}$Trackster [GeV]", fontsize=16)
ax.set_title("GNN SuperTrackster - noLC", fontsize=18)
# plt.savefig("EnergyVsScore_noLC.png")

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot()
# ax.scatter(scores_cand, candidate_energies, marker=".")
# ax.set_xlabel(r"Score (Energy $\cap$ over $\cup$)", fontsize=16)
# ax.set_ylabel(r"Energy of TICL Candidate [GeV]", fontsize=16)
# plt.savefig("EnergyVsScore_candidate_LC.png")

cmap = cm.get_cmap("inferno").copy()
cmap.set_bad(cmap(0))
# cmap = cm.inferno
# cmap.set_bad(cmap(0))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot()
h = ax.hist2d(scores, pred_energies, density=True, cmap=cmap, bins = 30, range = [[0., 1.],[0, max(pred_energies)]])
ax.set_xlabel(r"Score (Energy $\cap$ over $\cup$)", fontsize=16)
ax.set_ylabel(r"Energy of $\it{Super}$Trackster [GeV]", fontsize=16)
ax.set_title("GNN SuperTrackster", fontsize=18)
fig.colorbar(h[3], ax=ax)
# plt.savefig("EnergyVsScore_hist2d_LC.png")

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot()
h = ax.hist2d(scores_noLC, pred_energies_noLC, density=True, cmap=cmap, bins = 30, range = [[0., 1.],[0, max(pred_energies)]])
ax.set_xlabel(r"Score (Energy $\cap$ over $\cup$)", fontsize=16)
ax.set_ylabel(r"Energy of $\it{Super}$Trackster [GeV]", fontsize=16)
ax.set_title("GNN SuperTrackster - noLC", fontsize=18)
fig.colorbar(h[3], ax=ax)
# plt.savefig("EnergyVsScore_hist2d_noLC.png")

fig = plt.figure(figsize=(18, 8))
ax1 = fig.add_subplot(121)
h1 = ax1.hist2d(scores, pred_energies, norm=matplotlib.colors.LogNorm(), density=True, cmap=cmap, bins = 30, range = [[0., 1.],[0, max(pred_energies)]])
ax1.set_xlabel(r"Score (Energy $\cap$ over $\cup$)", fontsize=16)
ax1.set_ylabel(r"Energy of $\it{Super}$Trackster [GeV]", fontsize=16)
ax1.set_title("GNN SuperTrackster", fontsize=18)
fig.colorbar(h1[3], ax=ax1)
ax2 = fig.add_subplot(122)
h2 = ax2.hist2d(scores_noLC, pred_energies_noLC, norm=matplotlib.colors.LogNorm(), density=True, cmap=cmap, bins = 30, range = [[0., 1.],[0, max(pred_energies)]])
ax2.set_xlabel(r"Score (Energy $\cap$ over $\cup$)", fontsize=16)
# ax2.set_ylabel(r"Energy of $\it{Super}$Trackster [GeV]", fontsize=16)
ax2.set_title("GNN SuperTrackster - noLC", fontsize=18)
fig.colorbar(h2[3], ax=ax2)
plt.savefig("EnergyVsScore_hist2d_both.png")


# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot()
# h = ax.hist2d(scores_cand, candidate_energies, norm=matplotlib.colors.LogNorm(), cmap=cm.inferno, bins = 30, range = [[0., 1.],[0, max(candidate_energies)]])
# ax.set_xlabel(r"Score (Energy $\cap$ over $\cup$)", fontsize=16)
# ax.set_ylabel(r"Energy of TICL Candidate [GeV]", fontsize=16)
# fig.colorbar(h[3], ax=ax)
# plt.savefig("EnergyVsScore_candidate_hist2d_LC.png")

print(f"{more_truth_clusters} events with >2 truth clusters")
print(f"{isolated_truth_cluster} events with >2 truth clusters AND a truth cluster with one trackster")
print(f"{len(scores)} SuperTracksters")
print(f"{len(scores_noLC)} SuperTrackster - noLC")
# print(f"{len(scores_cand)} TICL Candidates")
print(f"{tot_clue3d_tracksters} CLUE3D tracksters")
print(f"{tot_truth_clusters} total sim tracksters from CP")
print(f"{events_processed} events")
