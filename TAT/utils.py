import networkx as nx
import numpy as np
import random
import torch
import os
import sys
import copy
import pickle
from scipy.sparse import csr_matrix
from copy import deepcopy
from typing import List, Union
from tqdm import tqdm
from itertools import combinations, permutations
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data.dataloader import Collater


def expand_edge_index_and_timestamps(edge_index: List, timestamps: List, set_indice: List) -> Union[np.array, np.array]:
    # test
    reserved_tedge_num = 20
    for i, ts in enumerate(timestamps):
        if len(ts) > reserved_tedge_num:
            timestamps[i] = np.sort(ts)[:reserved_tedge_num]

    edge_index = np.array(edge_index) # [2, |E|]
    num_temp_edges = np.sum( [len(x) for x in timestamps] )
    timestamps_expand = np.zeros(num_temp_edges, dtype=np.float)
    edge_index_expand = np.zeros((2, num_temp_edges), dtype=np.int)
    cumsum = 0
    for i in range( edge_index.shape[1] ):
        num_t = len( timestamps[i] )
        edge_index_expand[:, cumsum: cumsum+num_t ] = edge_index[:, i, np.newaxis].repeat(num_t, axis=-1)
        timestamps_expand[cumsum: cumsum+num_t] = np.array( timestamps[i] )
        cumsum += num_t

    timestamps_expand = timestamps_expand.max() - timestamps_expand 

    # normalize timestamps
    timestamps_expand = time_normalize(timestamps_expand)

    edge_index_expand = np.concatenate([edge_index_expand, edge_index_expand[ [1, 0], : ]], axis=1)
    timestamps_expand = np.concatenate([timestamps_expand, timestamps_expand], axis=0)

    # permute
    permutation = np.random.permutation(len(timestamps_expand))
    edge_index_expand = edge_index_expand[:, permutation]
    timestamps_expand = timestamps_expand[permutation]

    assert edge_index_expand.shape[1] == timestamps_expand.shape[0], 'length of expanded edge index must match length of expanded timestamps'
    return edge_index_expand, timestamps_expand


class GraphDataset(Dataset):
    def __init__(self, root_dir, filelist, sp_feature_type='sp', model='TAT'):
        self.root_dir = Path(root_dir)
        self.filelist = filelist 
        self.model = model
        self.sp_feature_type = sp_feature_type

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        with open( self.root_dir/self.filelist[index], 'rb' ) as f:
            data_obj = pickle.load(f)
        # import ipdb; ipdb.set_trace()
        edge_index = data_obj['edge_index']
        set_indice = data_obj['set_indice']
        sp_features = data_obj['sp_features']
        label = data_obj['label']
        timestamps = data_obj['timestamp'] 

        # features
        node_attribute_matrix = np.zeros( (sp_features.shape[0], 3) )
        node_attribute_matrix[set_indice[0], 0] = 1
        node_attribute_matrix[set_indice[1], 1] = 1
        node_attribute_matrix[set_indice[2], 2] = 1
        if self.model not in ['DE-GNN', 'TAT']:
            sp_features = np.zeros_like(sp_features)
        else:
            if self.sp_feature_type == 'sp': 
                pass
            elif self.sp_feature_type == 'rw': # random walk
                sp_features = get_rw_features(edge_index, sp_features.shape[0], set_indice)
            else:
                raise NotImplementedError

        node_attribute_matrix = np.concatenate([node_attribute_matrix, sp_features], axis=1) 
        edge_index_expanded, timestamps_expanded = expand_edge_index_and_timestamps(edge_index, timestamps, set_indice)

        # to tensors
        node_attribute_matrix = torch.FloatTensor(node_attribute_matrix) 
        edge_index_expanded = torch.LongTensor( edge_index_expanded )
        timestamps_expanded = torch.FloatTensor(timestamps_expanded)
        set_indice = torch.LongTensor(set_indice).unsqueeze(0)  
        y = torch.LongTensor([label])
        sample_G = Data(x=node_attribute_matrix, edge_index=edge_index_expanded, y=y, set_indice=set_indice, timestamps=timestamps_expanded)
        return sample_G

    def show_statistics(self, name):
        self.stat_array = np.zeros((6, ))
        self.mean_size = []
        self.num_tedges = []

        for filename in self.filelist:
            with open(self.root_dir/filename, 'rb') as f:
                data_obj = pickle.load(f)
                label = int(data_obj['label'])
                self.stat_array[label] += 1
                self.mean_size.append(data_obj['sp_features'].shape[0])
        self.mean_size = np.mean(self.mean_size)
        print('{} set: {}, mean size: {:.1f}'.format(name, self.stat_array, self.mean_size, ))

def get_rw_features(edge_index, N, set_indice, max_depth=2):
    epsilon = 1e-6
    adj = csr_matrix((np.ones(len(edge_index[0]),), (edge_index[0], edge_index[1])), shape=(N, N)) # construct sparse adj matrix
    adj = adj.toarray()
    adj = adj / (adj.sum(1, keepdims=True) + epsilon) # normalized adj matrix

    rw_features = [ np.eye(N)[set_indice], ]
    for i in range(max_depth):
        rw_features.append(np.matmul(rw_features[-1], adj))
    rw_features = np.stack(rw_features, axis=2)
    rw_features = np.sum(rw_features, axis=0)
    return rw_features

def load_raw_data(edgefile):
    """Edge list file to sparse matrix"""
    edgearray = np.loadtxt(edgefile)
    if edgearray.shape[1] == 3:
        edgearray = edgearray[:, :2]
    edgearray = edgearray.astype(np.int)

    g = nx.from_edgelist(edgearray)
    spmatrix = nx.to_scipy_sparse_matrix(g, format='csc')

    return spmatrix

def collect_tri_sets(G):
    """Enumerate all triangles on G, efficiency issue?
    """
    tri_sets = set(frozenset([node1, node2, node3]) for node1 in G for node2, node3 in combinations(G.neighbors(node1), 2) if G.has_edge(node2, node3))
    return [list(tri_set) for tri_set in tri_sets]

def collect_wedge_sets(G):
    """Enumerate all wedges"""

    wedge_sets = set(frozenset([node1, node2, node3]) for node1 in G for node2, node3 in  combinations(G.neighbors(node1), 2) if not G.has_edge(node2, node3))
    return [list(wedge_set) for wedge_set in wedge_sets]


def preprocess(edgearray):
    edges = {}
    if edgearray.shape[1] == 3: # has timestamp
        for (u, v, t) in edgearray:
            if not (u, v) in edges:
                edges[(u, v)] = t
    return edges

def time_normalize(time):
    max_t = time.max() + 1
    time = time * 3e6 / max_t
    return time

def read_file(dataset, args, pprocess=False, return_edgearray=False, normalize_time=False):
    # dataset = args.dataset
    # directed = args.directed

    directory = Path(args.datadir) / (dataset + '.txt')
    edgearray = np.loadtxt(directory)
    edges = edgearray[:, :2].astype(np.int)
    edgearray[:, -1] = edgearray[:, -1] - min(edgearray[:, -1])

    mask = edges[:, 0] != edges[:, 1] 
    edges = edges[mask]
    edgearray = edgearray[mask]
    edges = edges.tolist() # mush be a list, or an ajcanency np array

    # if directed:
    #     G = nx.DiGraph(edges)
    # else:
    G = nx.Graph(edges)

    if pprocess:
        edges = preprocess(edgearray) # a dict
        for edge, t in edges.items():
            G[edge[0]][edge[1]]['timestamp'] = t
            pass
    else:
        for i, edge in enumerate(edges): # a list
            if G[edge[0]][edge[1]].get('timestamp', None) is None:
                G[edge[0]][edge[1]]['timestamp'] = [edgearray[i][-1]]
            else:
                G[edge[0]][edge[1]]['timestamp'].append(edgearray[i][-1])

    # relabel all nodes using integers from 0 to N-1
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

    if return_edgearray:
        return G, edgearray
    else: return G


def split_datalist(data_list, train_num):
    train_set= data_list[:train_num]
    val_test_num = len(data_list) - train_num
    val_set = data_list[train_num:train_num+val_test_num//2]
    test_set = data_list[train_num+val_test_num//2:]

    return train_set, val_set, test_set

def determine_triad_class(G, set_indice):
    n1, n2, n3 = set_indice
    t_12 = min(G[n1][n2]['timestamp']) 
    t_13 = min(G[n1][n3]['timestamp'])
    t_23 = min(G[n2][n3]['timestamp'])

    times = list(sorted([t_12, t_13, t_23]))
    times = np.array(times).reshape((1, -1))

    permutations = np.array( [
        [t_12, t_13, t_23],
        [t_12, t_23, t_13],
        [t_13, t_12, t_23],
        [t_13, t_23, t_12],
        [t_23, t_12, t_13],
        [t_23, t_13, t_12],
        ] )
    index = np.argmax(np.all(permutations == times, axis=1) ) # 肯定有一个会是true
    return index


def sample_tri_wedge_neg_sets(G, data_usage=0.5):
    tri_samples = collect_tri_sets(G) # all triads on the graph, list
    wedge_samples = collect_wedge_sets(G) # all wedges on the graph, list

    tri_samples = np.array(tri_samples)
    wedge_samples = np.array(wedge_samples)

    # assert len(tri_samples.shape) == 2
    # assert len(wedge_samples.shape) == 2

    neg_samples = sample_neg_sets(G, n_samples=min(len(tri_samples), len(wedge_samples)) )

    if data_usage < 1 - 1e-5:
        tri_samples = retain_partial(tri_samples, ratio=data_usage)
        wedge_samples = retain_partial(wedge_samples, ratio=data_usage)
    print('# triads: {}, # wedges: {}'.format(len(tri_samples), len(wedge_samples)))

    return tri_samples, wedge_samples, neg_samples

def sample_neg_sets(G, n_samples):
    """ a negative sample is a tri-set, with only one edge. """
    neg_sets = []
    n_nodes = G.number_of_nodes()
    max_iter = 1e9
    count = 0
    while len(neg_sets) < n_samples:
        count += 1
        if count >= max_iter:
            raise Exception('Can not sample enough negative samples (tri-set with only one edge)')
        n1 = random.randint(0, n_nodes-1)
        n2 = random.choice(list(G.neighbors(n1)))
        n3 = random.randint(0, n_nodes-1) 
        if G.has_edge(n1, n3) + G.has_edge(n2, n3) == 0: 
            neg_sets.append([n1, n2, n3])

    return np.array(neg_sets)

def retain_partial(items, ratio):
    if type(ratio) is float and ratio < 1.0 + 1e-5: # ratio
        indices = np.random.choice(len(items), int(ratio * len(items)), replace=False)
    elif type(ratio) is int and ratio <= len(items): # number
        indices = np.random.choice(len(items), ratio, replace=False)
    return items[indices]

def get_mask(indices, length):
    mask = np.zeros(length, dtype=np.int)
    mask[indices] = 1
    return mask

def sort_set_indices_and_assign_labels(G, set_indices):

    argumented = False
    if argumented is True:
        set_indices_permute = []
        labels_permute = []
        for set_indice in set_indices:
            set_indice_p = random.sample(list(permutations(set_indice)), k=2)
            labels_p = [ determine_triad_class(G, s ) for s in set_indice_p ]
            set_indices_permute.extend(set_indice_p)
            labels_permute.extend(labels_p)

        set_indices = np.array(set_indices_permute)
        labels = np.array(labels_permute)

    else:
        labels = []
        for i, set_indice in enumerate(set_indices):
            label = determine_triad_class(G, set_indice)
            labels.append(label)
        labels = np.array(labels)

    return set_indices, labels

def permute(set_indices, labels):
    permutation = np.random.permutation(len(set_indices))
    set_indices = set_indices[permutation]
    labels = labels[permutation]
    return set_indices, labels

def generate_set_indices(G, test_ratio=0.2):
    """Generate set indeces, which are used for training/test target sets. But no labels now.
    # only triad prediction, 6 classes.
    """
    print('Generating train/test set indices and labels from graph...')

    tri_samples, wedge_samples, neg_samples = sample_tri_wedge_neg_sets(G)

    set_indices = tri_samples

    permutation = np.random.permutation(len(set_indices))
    set_indices = set_indices[permutation] # permute
    train_num = int(len(set_indices) * (1 - test_ratio) )
    set_indices_train, labels_train = sort_set_indices_and_assign_labels(G, set_indices[:train_num])
    set_indices_test, labels_test = sort_set_indices_and_assign_labels(G, set_indices[train_num:])

    set_indices_train, labels_train = permute(set_indices_train, labels_train)
    set_indices_test, labels_test = permute(set_indices_test, labels_test)

    set_indices = np.concatenate([set_indices_train, set_indices_test], axis=0)
    labels = np.concatenate([labels_train, labels_test], axis=0)
    train_num = len(set_indices_train)

    print('Generated {} train+test instances in total.'.format(len(set_indices)))

    return G, set_indices, labels, train_num


def get_hop_num(prop_depth, layers, max_sp):
    return int(prop_depth * layers)


def extract_subgraph(G, set_indices, labels, prop_depth, layers, max_sp, parallel, cached_dir):
    print("Extracting subgraphs and encoding...(parallel: {})".format(parallel))
    data_list = []
    hop_num = get_hop_num(prop_depth, layers, max_sp)

    if not parallel:
        for i, (set_indice, y) in tqdm(enumerate(zip(set_indices, labels)), total=len(set_indices)):
            data = get_data_sample(G, set_indice, y, hop_num, max_sp, cached_dir, i)
            data_list.append(data)
    else:
        raise NotImplementedError('not support parallel processing now')
    return data_list

def load_cached_datalist(cached_dir):
    if not isinstance(cached_dir, Path):
        cached_dir = Path(cached_dir)
    filelist = cached_dir.glob('*')
    filelist = list(filelist)
    random.shuffle(filelist)
    # import ipdb; ipdb.set_trace()
    data_list = filelist
    print("cached dataset {} loaded.".format(cached_dir.absolute()))
    return data_list


def save_datalist(data_list, args, logger):
    """
    data_list: [(edge_index, set_indice, label), ...]
    """
    logger.info("pickling ...")
    filedir = Path(args.datadir) / (args.dataset + '_cached.pk')
    with open(filedir, 'wb') as f:
        pickle.dump(data_list, f)
    logger.info("dataset {} cached at {}.".format(args.dataset, filedir))

def k_hop_neighbors(G, set_indice, k):
    """Return the neighbors in k-hop of the node(s)
    set_indices: the central node set
    """
    neighbors = set(copy.copy(set_indice))
    current_neighbors = set(copy.copy(set_indice))
    for i in range(k):
        next_hop_neighbors = []
        for n in current_neighbors:
            next_hop_neighbors.extend(list(G.neighbors(n)))
        next_hop_neighbors = set(next_hop_neighbors) - neighbors
        neighbors = neighbors.union(next_hop_neighbors)
        current_neighbors = next_hop_neighbors
    neighbors = list(neighbors)
    return neighbors

def onehot_encoding(indexs, n):
    """ convert integer indexs to one-hot encodings """
    assert max(indexs) <= n-1, "index out of range"
    eye_matrix = np.eye(n)
    return eye_matrix[indexs]

def has_edge_(G, edgetuple):
    return G.has_edge(edgetuple[0], edgetuple[1])

def edge_timestamp_(G, edgetuple):
    return G[edgetuple[0]][edgetuple[1]]['timestamp']

def remove_new_edges(sub_G, set_indice):
    latest_timestamp = []
    for (u, v) in combinations(set_indice, 2):
        if sub_G.has_edge(u, v):
            latest_timestamp.extend(sub_G[u][v]['timestamp'])
    # import ipdb; ipdb.set_trace()
    assert len(latest_timestamp) > 0
    latest_timestamp = np.min(latest_timestamp)
    assert isinstance(latest_timestamp, float), 'it must be a single float number, but got a {}'.format(type(latest_timestamp))

    edges_candidate = []
    for (u, v) in sub_G.edges():
        edge_timestamp = np.sort(sub_G[u][v]['timestamp'])
        if edge_timestamp[0] >= latest_timestamp:
            edges_candidate.append([u, v])
        else:
            mask = edge_timestamp < latest_timestamp
            edge_timestamp = edge_timestamp[mask]
            sub_G[u][v]['timestamp'] = edge_timestamp.tolist()
            assert isinstance(sub_G[u][v]['timestamp'], list)

    sub_G.remove_edges_from(edges_candidate)

    # to be a connected component
    reachable_nodes = [nx.descendants(sub_G, n) for n in set_indice] + [set(set_indice)]
    reachable_nodes = set.union(*reachable_nodes)
    sub_G = sub_G.subgraph(reachable_nodes).copy()

    return sub_G

def get_data_sample(G, set_indice, y, hop_num, max_sp, root_dir, findex):
    """ Extract a subgraph enclosing set_indice.
    """
    pid = os.getpid()

    # assert len(set_indice) == 3, "Currently only support 3-node set, e.g., wedges, triads"

    sub_nodes = k_hop_neighbors(G, set_indice, hop_num)
    sub_G = G.subgraph(sub_nodes).copy()

    sub_G = remove_new_edges(sub_G, set_indice)

    old_new_dict = dict( zip(list(sub_G.nodes), range(sub_G.number_of_nodes()) ) )
    sub_G = nx.convert_node_labels_to_integers(sub_G, first_label=0, ordering='default')
    set_indice = list(map(lambda x: old_new_dict[x], set_indice))
    edge_index = np.array(sub_G.edges, dtype=np.int).T.tolist()

    if len(edge_index) != 2 or sub_G.number_of_nodes() <= 5: # bad sample
        return None

    timestamps = [sub_G[u][v]['timestamp'] for (u, v) in zip(edge_index[0], edge_index[1]) ]
    sp_features = get_tde_features(sub_G, set_indice, max_sp)

    filename = '_'.join([str(pid), str(findex)])
    sample = {'edge_index': edge_index,
              'timestamp': timestamps,
              'set_indice': set_indice,
              'sp_features': sp_features,
              'label': y}

    with open(Path(root_dir)/filename, 'wb') as f:
        pickle.dump(sample, f)

    return filename

def preprocessing_cached_data(dataset, args, force_cache=True): # TODO: move to preprocessing
    """the main function to get train/val/test (temporal) subgraph data samples.
    """
    G = read_file(dataset, args)
    cached_dir = (Path(args.datadir)/args.dataset/'cached').absolute()

    if (not cached_dir.exists()) or force_cache:
        if cached_dir.exists():
            os.system('rm -rf {}'.format(cached_dir.absolute()))
            os.makedirs(cached_dir)
        else:
            os.makedirs(cached_dir)
        print('Cached directory {} newly created'.format(cached_dir.absolute()))
        print('generating data samples...')

        G, set_indices, labels, train_num = generate_set_indices(G, args.test_ratio)

        data_list = extract_subgraph(G, set_indices, labels, args.prop_depth, args.layers, args.max_sp,
                                    False, cached_dir)
    else:
        print('Cached samples already exist.')    
    

def load_dataloaders(G, args): # TODO: move to preprocessing
    """the main function to get train/val/test (temporal) subgraph data samples.
    """
    G = deepcopy(G)
    cached_dir = (Path(args.datadir)/args.dataset/'cached').absolute()

    assert cached_dir.exists(), "Cached samples directory must exist."
    data_list = load_cached_datalist(cached_dir)
    data_list = data_list[:int(len(data_list)*args.data_usage)] 
    train_num = int(len(data_list) * (1 - args.test_ratio))

    train_set, val_set, test_set = split_datalist(data_list, train_num)
    train_set = list(filter(None, train_set))
    val_set = list(filter(None, val_set))
    test_set = list(filter(None, test_set))
    assert len(train_set) > 0
    assert len(val_set) > 0
    assert len(test_set) > 0

    # Customed dataset
    train_set = GraphDataset(cached_dir, train_set, sp_feature_type=args.sp_feature_type, model=args.model )
    val_set = GraphDataset(cached_dir, val_set, sp_feature_type=args.sp_feature_type, model=args.model )
    test_set = GraphDataset(cached_dir, test_set, sp_feature_type=args.sp_feature_type, model=args.model )

    if args.debug:
        train_set.show_statistics('train')
        val_set.show_statistics('val')
        test_set.show_statistics('test')

    train_loader, val_loader, test_loader = make_dataloaders(train_set, val_set, test_set, args.batch_size)
    print("Train size: {}, val size: {}, test size: {}, val_test ratio: {}".format(len(train_set), len(val_set), len(test_set), args.test_ratio))

    feature_dim = train_set[0].x.shape[1]

    return (train_loader, val_loader, test_loader), feature_dim

# def get_data(G, args, logger): # TODO: move to preprocessing
#     """the main function to get train/val/test (temporal) subgraph data samples.
#     """
#     G = deepcopy(G)
#     cached_dir = (Path(args.datadir)/args.dataset/'cached').absolute()

#     if args.force_cache or not cached_dir.exists():
#         if cached_dir.exists():
#             os.system('rm -rf {}'.format(cached_dir.absolute()))
#             os.makedirs(cached_dir)
#         else:
#             os.makedirs(cached_dir)
#         logger.info('Cached directory {} newly created'.format(cached_dir.absolute()))
#         logger.info('generating fresh data samples...')

#         G, set_indices, labels, train_num = generate_set_indices(G, logger, args)

#         data_list = extract_subgraph(G, set_indices, labels, args.prop_depth, args.layers, args.max_sp,
#                                     args.parallel, cached_dir, args.debug, logger)
#     else:
#         assert cached_dir.exists(), "cached directory must exist"
#         data_list = load_cached_datalist(cached_dir, logger)
#         data_list = data_list[:int(len(data_list)*args.data_usage)] 
#         train_num = int(len(data_list) * (1 - args.test_ratio))

#     train_set, val_set, test_set = split_datalist(data_list, train_num)
#     train_set = list(filter(None, train_set))
#     val_set = list(filter(None, val_set))
#     test_set = list(filter(None, test_set))
#     assert len(train_set) > 0
#     assert len(val_set) > 0
#     assert len(test_set) > 0

#     # Customed dataset
#     train_set = GraphDataset(cached_dir, train_set, sp_feature_type=args.sp_feature_type, model=args.model )
#     val_set = GraphDataset(cached_dir, val_set, sp_feature_type=args.sp_feature_type, model=args.model )
#     test_set = GraphDataset(cached_dir, test_set, sp_feature_type=args.sp_feature_type, model=args.model )

#     if args.debug:
#         train_set.show_statistics('train', logger)
#         val_set.show_statistics('val', logger)
#         test_set.show_statistics('test', logger)

#     train_loader, val_loader, test_loader = make_dataloaders(train_set, val_set, test_set, args.batch_size)
#     logger.info("Train size: {}, val size: {}, test size: {}, val_test ratio: {}".format(len(train_set), len(val_set), len(test_set), args.test_ratio))

#     feature_dim = train_set[0].x.shape[1]

#     return (train_loader, val_loader, test_loader), feature_dim

def make_dataloaders(train_set, val_set, test_set, batch_size):
    # num_workers = 1
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=Collater(follow_batch=[], exclude_keys=[]))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=Collater(follow_batch=[], exclude_keys=[]))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=Collater(follow_batch=[], exclude_keys=[]))
    return train_loader, val_loader, test_loader


def set_random_seed(args):
    """ Set all seeds"""
    seed = args.seed
    # seed = random.randint(0, 100)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def estimate_storage(dataloaders, names, logger):
    total_gb = 0
    for dataloader, name in zip(dataloaders, names):
        dataset = dataloader.dataset
        storage = 0
        total_length = len(dataset)
        sample_size = 200
        for i in np.random.choice(total_length, sample_size):
            storage += (sys.getsizeof(dataset[i].x.storage()) + sys.getsizeof(dataset[i].edge_index.storage()) +
                        sys.getsizeof(dataset[i].y.storage()) + sys.getsizeof(dataset[i].set_indice.storage()) )
        gb = storage*total_length/sample_size/1e9
        total_gb += gb
    logger.info('Data roughly takes {:.4f} GB in total'.format(total_gb))
    return total_gb


def get_tde_features(G, set_indice, max_sp,):
    """ G: subgraph here
    set_indice: target triad set
    """
    dim = max_sp + 2
    set_size = len(set_indice)
    sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
    for i, node in enumerate(set_indice):
        for node_ngh, length in nx.shortest_path_length(G, source=node).items():
            sp_length[node_ngh, i] = length
    sp_length = np.minimum(sp_length, max_sp) 
    onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
    features_sp = onehot_encoding[sp_length].sum(axis=1) 

    tde_attribute_matrix = features_sp
    return tde_attribute_matrix

def timing_sort(G):
    """
    1, sort edges using timestamps
    2, assign edges with same timestamps same ranks
    """
    # sort first
    sorted_edges = sorted(G.edges, key=lambda e: G[e[0]][e[1]]['timestamp'], reverse=True) 
    if len(sorted_edges) == 0:
        return [], []
    # then ergodic the results
    rank_list = []
    rank = 0
    rank_list.append(rank)
    previous_value = G[sorted_edges[0][0]][sorted_edges[0][1]]['timestamp']
    for i, edge in enumerate(sorted_edges[1:]):
        current_value = G[edge[0]][edge[1]]['timestamp']

        if current_value != previous_value:
            rank += 1

        rank_list.append(rank)
        previous_value = current_value
    return sorted_edges, rank_list



