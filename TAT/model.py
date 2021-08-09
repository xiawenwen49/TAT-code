import numpy as np
import torch
import torch.nn as nn
from itertools import combinations
from torch import Tensor
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, TAGConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from typing import Union, Tuple, List, Optional, Dict

class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(), dropout=0):
        super(FeedForwardNetwork, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features), self.act, self.dropout)
        self.layer2 = nn.Sequential(nn.Linear(in_features, out_features) )

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x

class GNNModel(nn.Module):
    def __init__(self, layers, in_features, hidden_features, out_features, prop_depth, dropout, model_name,
                 time_encoder_type: str, time_encoder_maxt: float, time_encoder_rows: int, time_encoder_dimension: int,
                 time_encoder_discrete: str,):
        super(GNNModel, self).__init__()
        self.layers = layers
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.prop_depth = prop_depth
        self.model_name = model_name

        Layer = self.get_layer_class()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        if self.model_name in ['DE-GNN', 'TAGCN']: 
            self.layers.append(Layer(in_channels=in_features, out_channels=hidden_features, K=prop_depth))
        elif self.model_name == 'GIN': 
            raise NotImplementedError
        else: # GCN, GAT, GraphSAGE
            self.layers.append(Layer(in_channels=in_features, out_channels=hidden_features))
        
        if layers > 1:
            for i in range(layers - 1):
                if self.model_name in ['DE-GNN', 'TAGCN']:
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features, K=prop_depth))
                elif self.model_name == 'GAT':
                    if i == layers - 2: 
                        heads = 1
                    else: heads = 8
                    in_channels = self.layers[-1].heads*self.layers[-1].out_channels
                    self.layers.append(Layer(in_channels=in_channels, out_channels=hidden_features, heads=heads))
                elif self.model_name == 'GIN':
                    raise NotImplementedError
                else:
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features))

        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_features) for i in range(layers)])
        self.merger = nn.Linear(3 * hidden_features, hidden_features)

        self.feed_forward = FeedForwardNetwork(hidden_features, out_features)

        t_args = {} # time encoder args
        t_args['maxt'] = time_encoder_maxt
        t_args['rows'] = time_encoder_rows
        t_args['dimension'] = time_encoder_dimension
        t_args['discrete'] = time_encoder_discrete

        if time_encoder_type == 'tat':
            self.time_encoder = TATEncoder(**t_args)
        elif time_encoder_type == 'harmonic':
            t_args = {}
            self.time_encoder = HarmonicEncoder(**t_args)
        elif time_encoder_type == 'empty':
            self.time_encoder = EmptyEncoder(**t_args)
        else:
            raise ValueError("Not supported time encoder scheme")

    
    def forward(self, batch):
        x = batch.x

        edge_index = batch.edge_index
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        
        x = self.get_mini_batch_embeddings(x, batch)
        x = self.feed_forward(x)

        return x

    def get_layer_class(self, ):
        layer_dict = {'DE-GNN': TAGConv, 'TAGCN': TAGConv, 'GIN': GINConv, 'GCN': GCNConv, 'GraphSAGE': SAGEConv, 'GAT': GATConv}
        Layer = layer_dict.get(self.model_name, None)
        if Layer is None:
            raise NotImplementedError("Not implemented model: {}".format(self.model_name))
        return Layer

    
    def get_mini_batch_embeddings(self, x, batch): # for whole graph embedding
        device = x.device
        set_indice, batch_idx, num_graphs = batch.set_indice, batch.batch, batch.num_graphs
        num_nodes = torch.eye(num_graphs)[batch_idx].to(device).sum(dim=0) 
        zero = torch.tensor([0], dtype=torch.long).to(device)
        index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1] ]) 
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indice.size(-1)) 
        set_indice_batch = index_bases + set_indice 
        x = x[set_indice_batch] 
        x = self.pool(x) 
        return x

    def pool(self, x):
        """ readout 
        """
        x = self.merger(x.reshape((x.shape[0], -1))) 
        return x


def get_model(args, logger):
    if args.model in ['DE-GNN', 'TAGCN', 'GIN', 'GCN', 'GraphSAGE', 'GAT']:
        model = GNNModel(layers=args.layers, in_features=args.in_features, hidden_features=args.hidden_features, 
                        out_features=args.out_features, prop_depth=args.prop_depth, dropout=args.dropout, model_name=args.model,
                        time_encoder_type=args.time_encoder_type, time_encoder_maxt=args.time_encoder_maxt, time_encoder_rows=args.time_encoder_rows, time_encoder_dimension=args.time_encoder_dimension, time_encoder_discrete=args.time_encoder_discrete,)
    elif args.model in ['TAT']:
        model = TATModel(model_name=args.model, time_encoder_type=args.time_encoder_type, time_encoder_maxt=args.time_encoder_maxt,
                         time_encoder_rows=args.time_encoder_rows, time_encoder_dimension=args.time_encoder_dimension, time_encoder_discrete=args.time_encoder_discrete,
                         layers=args.layers, in_features=args.in_features, hidden_features=args.hidden_features, out_features=args.out_features,
                         dropout=args.dropout, negative_slope=args.negative_slope, set_indice_length=args.set_indice_length, attention=args.use_attention,)
    else:
        raise NotImplementedError
    return model


class TimeEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TimeEncoder, self).__init__()
        pass

class HarmonicEncoder(TimeEncoder):
    """ In paper 'Inductive representation learning on temporal graphs'
    """
    def __init__(self, dimension, factor=5, **kwargs):
        super(HarmonicEncoder, self).__init__()
        self.dimension = dimension
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))).float())
        self.phase = torch.nn.Parameter(torch.zeros(dimension).float())
    
    def forward(self, ts):
        # ts: [N, L]
        device = ts.device
        self.basis_freq = self.basis_freq.to(device)
        self.phase = self.phase.to(device)

        batch_size = ts.size(0)
        seq_len = 1
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        harmonic = harmonic.squeeze(1)

        return harmonic #self.dense(harmonic)

class EmptyEncoder(TimeEncoder):
    def __init__(self, maxt, mint: float = 0, rows: int = 50000, dimension: int = 128, discrete: str = 'uniform', **kwargs):
        super(EmptyEncoder, self).__init__()
        self.dimension = dimension
        pass

    def __call__(self, timestamps: Tensor):
        device = timestamps.device
        zeros = torch.zeros((timestamps.shape[0], self.dimension)).to(device)
        return zeros


class TATEncoder(TimeEncoder):
    """
    discrete + position encoding 
    """
    def __init__(self, maxt, mint: float = 0, rows: int = 50000, dimension: int = 128, discrete: str = 'uniform', **kwargs):
        super(TATEncoder, self).__init__()
        self.mint = mint
        self.maxt = maxt
        self.rows = rows
        self.deltat = maxt/rows
        self.dimension = dimension
        self.discrete = discrete

        timing_encoding_matrix = self.get_timing_encoding_matrix(rows, dimension)
        self.timing_encoding_matrix = timing_encoding_matrix

    def forward(self, timestamps: Tensor):
        device = timestamps.device
        if self.discrete == 'uniform':
            indexs = self.timestamps_to_indexs(timestamps) # np array
        elif self.discrete == 'log':
            indexs =self.timestamps_to_indexs_log(timestamps) # np array
        else:
            raise NotImplementedError

        return torch.FloatTensor(self.timing_encoding_matrix[indexs]).to(device) # to gpu tensor
        
    def timestamps_to_indexs_log(self, timestamps: Tensor):
        timestamps = timestamps.to('cpu').numpy()
        timestamps = np.clip(timestamps, 1, self.maxt-1).astype(np.float)
        indexs = (np.log(timestamps)/np.log(self.maxt)*self.rows).astype(np.int)
        return indexs

    def timestamps_to_indexs(self, timestamps: Tensor):
        """ convert float tensor timestamps to long tensor indexs 
        """
        timestamps = timestamps.to('cpu').numpy()
        indexs = (timestamps // self.deltat).astype(np.int)
        indexs = np.clip(indexs, 0, self.rows-1).astype(np.int)
        return indexs

    def get_timing_encoding_matrix(self, length, dimension, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
        """ 
        https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
        """
        assert dimension % 2 == 0, 'TAT time encoding dimension must be even'
        T = np.arange(length).reshape((length, 1))
        W_inv_log = np.arange(dimension//2)*2/(dimension-2) * np.log(max_timescale) 
        W = 1 / np.exp( W_inv_log )

        position_encodings = T @ W.reshape((1, dimension//2))
        position_encodings = np.concatenate([ np.sin(position_encodings), np.cos(position_encodings) ], axis=1)
        position_encodings = torch.Tensor(position_encodings) # [rows, dimension] torch tensor

        return position_encodings

class TATModel(nn.Module):
    """
    main TAT model
    """
    def __init__(self, model_name: str, time_encoder_type: str, time_encoder_maxt: float, time_encoder_rows: int, time_encoder_dimension: int,
                 time_encoder_discrete: str, layers: int, in_features: int, hidden_features: int, out_features: int, 
                 dropout: float = 0.0, negative_slope: float = 0.2, set_indice_length: int = 3, attention: bool = True ):
        """ 
        Args:
            model_name: 'TAT', ...
            time_encoder_type: 'tat', 'harmonic', 'empty'
        """
        super(TATModel, self).__init__()
        self.model_name = model_name
        self.time_encoder_type = time_encoder_type
        self.layers = layers
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.attention = attention
        
        t_args = {} # time encoder args
        t_args['maxt'] = time_encoder_maxt
        t_args['rows'] = time_encoder_rows
        t_args['dimension'] = time_encoder_dimension
        t_args['discrete'] = time_encoder_discrete

        if time_encoder_type == 'tat':
            self.time_encoder = TATEncoder(**t_args)
        elif time_encoder_type == 'harmonic':
            self.time_encoder = HarmonicEncoder(**t_args)
        elif time_encoder_type == 'empty':
            self.time_encoder = EmptyEncoder(**t_args)
        else:
            raise ValueError("Not supported time encoder scheme")

        self.act = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()

        if model_name == 'TAT':
            for i in range(layers):
                in_channels_ = (in_features + self.time_encoder.dimension)  if i == 0 else (self.layers[-1].out_channels * self.layers[-1].heads + self.time_encoder.dimension)
                heads_ = 4 if i < layers - 1 else 1
                self.layers.append( TATConv(self.time_encoder, in_channels=in_channels_, out_channels=hidden_features, heads=heads_, layer=i, attention=self.attention) )
        elif model_name == 'GAT':
            pass
        else: raise NotImplementedError

        self.feed_forward = nn.Sequential(
            nn.Linear(set_indice_length*hidden_features, hidden_features,), 
            nn.LeakyReLU(negative_slope=negative_slope), 
            nn.Linear(hidden_features, out_features) 
        ) # feed forward network for output scores

    def get_mini_batch_embedings(self, x, batch):
        device = x.device
        set_indice, batch_idx, num_graphs = batch.set_indice, batch.batch, batch.num_graphs
        num_nodes = torch.eye(num_graphs)[batch_idx].to(device).sum(dim=0) 
        zero = torch.tensor([0], dtype=torch.long).to(device)
        index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1] ]) 
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indice.size(-1)) 
        set_indice_batch = index_bases + set_indice # e.g., [64, 3] 
        x = x[set_indice_batch] # e.g., [64, 3, 100] 
        return x
    
    def pool(self, x):
        x = x.reshape((x.shape[0], -1)) # simply concat
        return x

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        timestamps = batch.timestamps

        # preprocessing edge_index and timestamps
        assert edge_index.shape[1] == timestamps.shape[0], 'length of edge_index must match length of timestamps'
        
        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, timestamps=timestamps)
            x = self.act(x)
            x = self.dropout(x)
        
        x = self.get_mini_batch_embedings(x, batch)
        merged_x = self.pool(x) # e.g., [63, 3, 100]
        scores = self.feed_forward(merged_x) #, e.g., [64, 300]
        return scores


class TATConv(MessagePassing):
    """ from MessagePassing class in torch_geometric
    """
    def __init__(self, time_encoder: TimeEncoder, in_channels: int, out_channels: int, 
                 heads: int = 1, layer: int = 0, concat: bool=True, negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = False, bias: bool = True, attention: bool = True, **kwargs):
        super(TATConv, self).__init__(aggr='add', node_dim=0, **kwargs) 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.layer = layer
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.time_encoder = time_encoder 
        self.attention = attention

        # gat attention scheme
        self.lin_l = Linear(in_channels, heads*out_channels, bias=False)
        self.lin_r = self.lin_l
        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        # query-key scheme
        self.lin_Q = Linear(in_channels, heads*out_channels, bias=False)
        self.lin_K = Linear(in_channels, heads*out_channels, bias=False)
        self.lin_V = Linear(in_channels, heads*out_channels, bias=False) 

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads*out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_Q.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)

        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)
    
    def forward(self, x, edge_index, timestamps, return_attention_weights=None):
        H, C = self.heads, self.out_channels
        self.timestamps_ = timestamps

        x_l = x_r = x
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                N = x_l.size(0)
                loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
                loop_index = loop_index.unsqueeze(0).repeat(2, 1)
                edge_index = torch.cat([edge_index, loop_index], dim=1)
                zeros = torch.zeros(N,  dtype=torch.long, device=timestamps.device)
                timestamps = torch.cat([timestamps, zeros], dim=0)
                self.timestamps_ = timestamps

        out = self.propagate(edge_index, x=(x_l, x_r))
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads*self.out_channels) # [N, heads*out_channels]
        else:
            out = out.mean(dim=1) # [N, out_channels]
        
        if self.bias is not None:
            out += self.bias
        
        if return_attention_weights is True:
            assert self.alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, self.alpha)
            else: raise ValueError("edge_index must be a Tensor, other dtypes are not supported now")
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, index: Tensor) -> Tensor:
        H, C = self.heads, self.out_channels
        if self.layer >= 0:
            time_encoding = self.time_encoder(self.timestamps_)
            zero_encoding = self.time_encoder(torch.zeros_like(self.timestamps_).to(self.timestamps_.device))
            # add time encoding
            x_j = torch.cat([x_j, time_encoding], axis=1)
            x_i = torch.cat([x_i, zero_encoding], axis=1)

        x_j = self.lin_l(x_j).view(-1, H, C)
        x_i = self.lin_l(x_i).view(-1, H, C)
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)

        if self.attention is False:
            alpha_j = alpha_i = torch.ones_like(alpha_i)

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i 
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ) 
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1) 
    
    def __repr(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)