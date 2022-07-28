import resource
import glob
import pickle

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