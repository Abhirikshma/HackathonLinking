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

def loadData(path, num_files = 1):
    f_edges_label = glob.glob(path + "*edges_labels.pkl" )
    f_edges = glob.glob(path + "*edges.pkl" )
    f_nodes_features = glob.glob(path + "*node_features.pkl" )
    edges_label = []
    edges = []
    nodes_features = []

    for i_f, _ in enumerate(f_edges_label):
        if(i_f <= num_files):
            f = f_edges_label[i_f]
            with open(f, 'rb') as fb:
                edges_label.append(pickle.load(fb))
            f = f_edges[i_f]
            with open(f, 'rb') as fb:
                edges.append(pickle.load(fb))
            f = f_nodes_features[i_f]
            with open(f, 'rb') as fb:
                nodes_features.append(pickle.load(fb))
        else:
            break
    return edges_label, edges, nodes_features