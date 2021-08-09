
import torch
import numpy as np
import copy
import pandas as pd
import os
import scipy
from torch import Tensor
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
# from xww.utils.tensorboard import TensorboardSummarizer
from .exp_study import rank_acc
from typing import List, Dict

criterion = torch.nn.functional.cross_entropy

def get_device(gpu_index):
    if gpu_index >= 0:
        assert torch.cuda.is_available(), 'cuda not available'
        assert gpu_index >= 0 and gpu_index < torch.cuda.device_count(), 'gpu index out of range'
        return torch.device('cuda:{}'.format(gpu_index))
    else:
        return torch.device('cpu')

def get_optimizer(model, args):
    optim = args.optimizer
    if optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise NotImplementedError


def train_model(model, dataloaders, args, logger):
    device = get_device(args.gpu)
    model = model.to(device)
    target_model = copy.deepcopy(model)
    target_model = target_model.to(device)

    train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args)
    metrics = {'loss': loss_metric, 'acc': acc_metric, 'auc': roc_auc_metric, 'macro_f1': f1_score_metric, 'ranked_acc': ranked_acc_metric}
    
    recorder = Recorder(minmax={'loss': 0, 'acc': 1, 'auc': 1, 'macro_f1': 1, 'ranked_acc': 1}, checkpoint_dir=args.checkpoint_dir, time_str=args.time_str)
    log_dir = Path(args.log_dir)/'tb_logs'/args.dataset/'_'.join([args.time_str, args.desc.replace(' ', '_')]) # add desc after time, as log dir name
    # summary_writer = TensorboardSummarizer(log_dir, hparams_dict=vars(args), desc=args.desc)
    summary_writer = None
    for step in range(args.epoch):
        optimize_model(model, target_model, train_loader, optimizer, logger, summary_writer, step, perm_loss=args.perm_loss)

        train_metrics_results = eval_model(model, train_loader, metrics, desc='train_eval', start_step=step*len(train_loader))
        val_metrics_results = eval_model(model, val_loader, metrics, desc='val_eval', start_step=step*len(val_loader))
        test_metrics_results = eval_model(model, test_loader, metrics, desc='test_eval', start_step=step*len(test_loader))

       
        recorder.append_full_metrics(train_metrics_results, 'train')
        recorder.append_full_metrics(val_metrics_results, 'val')
        recorder.append_full_metrics(test_metrics_results, 'test')
        recorder.append_model_state(model.state_dict(), metrics)

        test_best_metric, _ = recorder.get_best_metric('test')
        train_latest_metric = recorder.get_latest_metric('train')
        val_latest_metric = recorder.get_latest_metric('val')
        test_latest_metric = recorder.get_latest_metric('test')
        
        logger.info('epoch {}, best test acc/r_acc: {:.4f}/{:.4f}, train loss/acc/r_acc: {:.4f}/{:.4f}/{:.4f}, val acc/r_acc: {:.4f}/{:.4f}, test acc/r_acc: {:.4f}/{:.4f}'.format(
                    step, test_best_metric['acc'], test_best_metric['ranked_acc'], train_latest_metric['loss'], train_latest_metric['acc'], train_latest_metric['ranked_acc'],
                    val_latest_metric['acc'], val_latest_metric['ranked_acc'], test_latest_metric['acc'], test_latest_metric['ranked_acc'] ))
        
        recorder.save()
        recorder.save_model()

    # summary_writer.close()
    # overall statistics
    train_best_metric, train_best_epoch = recorder.get_best_metric('train')
    val_best_metric, val_best_epoch = recorder.get_best_metric('val')
    test_best_metric, test_best_epoch = recorder.get_best_metric('test')
    logger.info('best train acc/r_acc/auc/f1: {:.4f}/{:.4f}/{:.4f}/{:.4f}, epoch: {}/{}/{}/{}'.format( train_best_metric['acc'], train_best_metric['ranked_acc'], train_best_metric['auc'], train_best_metric['macro_f1'],
                                                                                                       train_best_epoch['acc'], train_best_epoch['ranked_acc'], train_best_epoch['auc'], train_best_epoch['macro_f1']))
    logger.info('best val acc/r_acc/auc/f1: {:.4f}/{:.4f}/{:.4f}/{:.4f}, epoch: {}/{}/{}/{}'.format( val_best_metric['acc'], val_best_metric['ranked_acc'], val_best_metric['auc'], val_best_metric['macro_f1'],
                                                                                                       val_best_epoch['acc'], val_best_epoch['ranked_acc'], val_best_epoch['auc'], val_best_epoch['macro_f1']))
    logger.info('best test acc/r_acc/auc/f1: {:.4f}/{:.4f}/{:.4f}/{:.4f}, epoch: {}/{}/{}/{}'.format( test_best_metric['acc'], test_best_metric['ranked_acc'], test_best_metric['auc'], test_best_metric['macro_f1'],
                                                                                                       test_best_epoch['acc'], test_best_epoch['ranked_acc'], test_best_epoch['auc'], test_best_epoch['macro_f1']))
    
    recorder.save()
    recorder.save_model()
    logger.info(f'record saved at {recorder.checkpoint_dir/recorder.time_str}')
    return recorder


def each_class_acc(predictions, labels):
    num_samples = np.zeros((5,), )
    num_correct = np.zeros((5,), )
    predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    for i, (p, l) in enumerate(zip(predictions, labels)):
        num_samples[l] += 1
        if p == l:
            num_correct[l] += 1
    
    class_acc = num_correct / num_samples
    print(class_acc)

def model_device(model):
    """ return device of a model
    """
    return next(model.parameters()).device


def eval_model(model, dataloader, metrics, **kwargs):

    device = model_device(model)
    model.eval()
    predictions = []
    labels = []
    desc = kwargs.get('desc', 'desc')

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=desc):
            try:
                batch = batch.to(device)
                prediction = model(batch)
                predictions.append(prediction)
                labels.append(batch.y)
            except RuntimeError:
                continue
            
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    metrics_results = compute_metric(predictions, labels, metrics)
    return metrics_results


def acc_metric(predictions, labels):
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    acc = (np.argmax(predictions, axis=1) == labels).sum() / labels.shape[0]
    return acc

def roc_auc_metric(predictions, labels):
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    predictions = scipy.special.softmax(predictions, axis=1)
    multi_class = 'ovr'
    if predictions.shape[1] == 2:
        predictions = predictions[:, 1]
        multi_class = 'raise'
    try:
        auc = roc_auc_score(labels, predictions, multi_class=multi_class)
    except ValueError:
        auc = 0
    return auc

def f1_score_metric(predictions, labels):
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    predictions = np.argmax(predictions, axis=1)
    macro_f1_score = f1_score(predictions, labels, average='macro')
    return macro_f1_score

def ranked_acc_metric(predictions, labels, rank=2):
    """ ACC@2
    """
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    class_rank = np.argsort( -1*predictions, axis=-1)
    labels = np.repeat(labels.reshape((-1, 1)), rank, axis=1)
    correct_predictions = np.max(class_rank[:, :rank] == labels, axis=1).flatten()
    acc = correct_predictions.sum()/labels.shape[0]
    return acc

def loss_metric(predictions, labels):
    """ cross entropy loss """
    if not isinstance(predictions, Tensor):
        predictions = torch.Tensor(predictions).cpu()
    if not isinstance(labels, Tensor):
        labels = torch.LongTensor(labels)
        labels = labels.to(predictions.device)

    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(predictions, labels, reduction='mean').item()
    return loss
    
def compute_metric(predictions, labels, metrics):
    metrics_results = {}
    for key, f in metrics.items():
        metrics_results[key] = f(predictions, labels)    
    return metrics_results

def permute_batch(batch, permute_idx, permute_rand):
    
    device = batch.set_indice.device
    # index bases
    set_indice, batch_idx, num_graphs = batch.set_indice, batch.batch, batch.num_graphs
    num_nodes = torch.eye(num_graphs)[batch_idx].to(device).sum(dim=0)
    zero = torch.tensor([0], dtype=torch.long).to(device)
    index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1] ])
    # reset x
    for i in range(num_graphs):
        batch.x[index_bases[i] + set_indice[i][0] ][:3] = 0
        batch.x[index_bases[i] + set_indice[i][1] ][:3] = 0
        batch.x[index_bases[i] + set_indice[i][2] ][:3] = 0
    
    # permute set_indice
    permute_idx = torch.LongTensor(permute_idx).to(device)
    for i in range(num_graphs):
        batch.set_indice[i] = batch.set_indice[i][ permute_idx[permute_rand[i]] ]
    
    # renew x
    new_set_indice = batch.set_indice
    for i in range(num_graphs):
        batch.x[index_bases[i] + new_set_indice[i][0] ][0] = 1 
        batch.x[index_bases[i] + new_set_indice[i][1] ][1] = 1 
        batch.x[index_bases[i] + new_set_indice[i][2] ][2] = 1

    return batch

def determine_triad_class_(set_indice, timestamp):
    n1, n2, n3 = set_indice
    t_12 = timestamp[(n1, n2)]
    t_13 = timestamp[(n1, n3)]
    t_23 = timestamp[(n2, n3)]

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
    index = np.argmax(np.all(permutations == times, axis=1) )
    return index

def time_dict(set_indice, y):
    assert np.all(set_indice == np.array([1, 2, 3], dtype=np.int) )
    times = {}
    if y == 0:
        times[(1, 2)] = 0 # time label
        times[(1, 3)] = 1
        times[(2, 3)] = 2
    elif y == 1:
        times[(1, 2)] = 0
        times[(2, 3)] = 1
        times[(1, 3)] = 2
    elif y == 2:
        times[(1, 3)] = 0
        times[(1, 2)] = 1
        times[(2, 3)] = 2
    elif y == 3:
        times[(1, 3)] = 0
        times[(2, 3)] = 1
        times[(1, 2)] = 2
    elif y == 4:
        times[(2, 3)] = 0
        times[(1, 2)] = 1
        times[(1, 3)] = 2
    elif y == 5:
        times[(2, 3)] = 0
        times[(1, 3)] = 1
        times[(1, 2)] = 2

    for k in list(times.keys()):
        times[(k[1], k[0])] = times[k]
    return times

def determine_permute_matrix():
    """
    permuted_label: 
                    array([ [0., 2., 1., 3., 4., 5.],
                            [1., 3., 0., 2., 5., 4.],
                            [2., 0., 4., 5., 1., 3.],
                            [3., 1., 5., 4., 0., 2.],
                            [4., 5., 2., 0., 3., 1.],
                            [5., 4., 3., 1., 2., 0.]])
    """
    permuted_label = np.zeros((6, 6)) 
    permute_idx = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0]
    ], dtype=np.int) 

    set_indice = np.array([1, 2, 3], dtype=np.int) 
    for y in range(6):
        times = time_dict(set_indice, y) 
        for i in range(6):
            pemu_idx = permute_idx[i]
            pemu_set_indice = set_indice[pemu_idx]
            pemu_y = determine_triad_class_(pemu_set_indice, times)
            permuted_label[y][i] = pemu_y
        
    inverse_permute_matrix = np.zeros_like(permuted_label, dtype=np.int)
    for i in range(6):
        inverse_permute_matrix[:, i] = np.argsort(permuted_label[:, i])

    return permute_idx, permuted_label, inverse_permute_matrix
    

def permute_optimize(model, prediction, target_model, batch, optimizer):
    """ for permutation loss training。针对一个batch。
    """
    device = model_device(model)
    mse = torch.nn.MSELoss()

    # permuted batch
    permute_idx, permuted_label, inverse_permute_matrix = determine_permute_matrix()
    num_graphs = batch.num_graphs
    pemu_rand = np.random.randint(0, 6, size=num_graphs)
    batch_copy = copy.deepcopy(batch) 
    permuted_batch = permute_batch(batch_copy, permute_idx, pemu_rand) # permuted batch

    # build target values
    target_model = target_model.to(device)
    with torch.no_grad():
        permuted_prediction = target_model(permuted_batch)
    
    inverse_permute_matrix = torch.LongTensor(inverse_permute_matrix).to(device)
    pemu_rand = torch.LongTensor(pemu_rand).to(device)
    for i in range(num_graphs): # align
        permuted_prediction[i] = permuted_prediction[i][inverse_permute_matrix[:, pemu_rand[i]]] # aligned with prediction
    permuted_prediction = permuted_prediction.detach() # pure tensor, no grad required

    # compute loss
    loss = mse(prediction, permuted_prediction)
    return loss


def optimize_model(model, target_model, dataloader, optimizer, logger, summary_writer, epoch, **kwargs):
    """ training for one epoch """
    model.train()
    device = model_device(model)
    passed_batches = 0
    update_batch = 16
    count = 0
    optimizer.zero_grad()
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='train'):
        try:
            batch = batch.to(device)
            batch_copy = copy.deepcopy(batch) # 
            model = model.to(device)
            label = batch.y
            prediction = model(batch)

            if epoch >= 10:
                weight = kwargs['perm_loss']
                ce_loss = criterion(prediction, label, reduction='mean')
                p_loss = permute_optimize(model, prediction, target_model, batch_copy, optimizer)
                loss = ce_loss + p_loss * weight
            else:
                loss = criterion(prediction, label, reduction='mean')
            loss.backward()

            if count >= update_batch:
                optimizer.step()
                optimizer.zero_grad()
                count = 0
            else:
                count += len(batch)
            
        except Exception as e: 
            if 'CUDA out of memory' in e.args[0]:
                logger.info(f'CUDA out of memory for batch {i}, skipped.')
                passed_batches += 1
            else: 
                raise
    # update target model
    target_model.load_state_dict(model.state_dict())
    logger.info('Passsed batches: {}/{}'.format(passed_batches, len(dataloader)))


class Recorder(object):
    def __init__(self, minmax: Dict, checkpoint_dir: str, time_str: str):
        """ recordes in checkpoint_dir/time_str/record.csv
        """
        self.minmax = minmax
        self.full_metrics = OrderedDict( {'train': [], 'val': [], 'test': []} )
        self.time_str = time_str

        if not isinstance(checkpoint_dir, Path):
            self.checkpoint_dir = Path(checkpoint_dir)
        else: self.checkpoint_dir = checkpoint_dir        
        
        if not (self.checkpoint_dir/self.time_str).exists():
            os.makedirs((self.checkpoint_dir/self.time_str))

    def append_full_metrics(self, metrics_results, name):
        assert name in ['train', 'val', 'test']
        self.full_metrics[name].append(metrics_results)

    def save(self):
        full_metrics = copy.deepcopy(self.full_metrics)

        filename = self.checkpoint_dir/self.time_str/'record.csv'
        for key in list(full_metrics.keys()):
            full_metrics[key] = pd.DataFrame(full_metrics[key])

        for key in list(full_metrics.keys()):
            full_metrics[key] = full_metrics[key].rename(columns=lambda x:key+'_'+x)

        df = pd.concat( list(full_metrics.values()), axis=1 )
        df.to_csv(filename, float_format='%.4f', index=True, index_label='epoch' )


    @classmethod
    def load(cls, checkpoint_dir, time_str):
        filename = Path(checkpoint_dir)/time_str/'record.csv'
        assert filename.exists, 'no such file: {}'.format(filename)
        df = pd.read_csv(filename)
        return df

    def get_best_metric(self, name):
        """
        return best value and best epoch
        name: 'train', 'val', 'test'
        """
        df = pd.DataFrame( self.full_metrics[name] )
        best_metric = {}
        best_epoch = {}
        for key in df.keys():
            data = np.array(df[key])
            best_metric[key] = np.max(data) if self.minmax[key] else np.min(data)
            best_epoch[key] = np.argmax(data) if self.minmax[key] else np.argmin(data)
        return best_metric, best_epoch
    
    def get_latest_metric(self, name):
        """
        name: train, val, test
        """
        latest_metric = self.full_metrics[name][-1]
        return latest_metric

    def append_model_state(self, state_dict, metrics):
        try:
            self.model_state.append(state_dict)
        except AttributeError:
            self.model_state = []
            self.model_state.append(state_dict)

    def save_model(self):
        try:
            best_metric, best_epoch = self.get_best_metric('test')
            for key, ep in best_epoch.items():
                torch.save( self.model_state[ep], self.checkpoint_dir/self.time_str/f'{key}_state_dict')
        except AttributeError:
            pass

    def load_model(self, checkpoint_dir, time_str):
        if not isinstance(checkpoint_dir, Path):
            checkpoint_dir = Path(checkpoint_dir)
        model_state_dict = {}
        state_dict_files = checkpoint_dir.glob('*_state_dict')
        for file in state_dict_files:
            model_state_dict[str(file).rstrip('_state_dict')] = torch.load(checkpoint_dir/time_str/file)
        return model_state_dict