import numpy as np
import torch
from itertools import combinations
from torch import Tensor


def count_triad_classes(G, set_indices):
    stat_array = np.zeros((6, ))
    for set_indice in set_indices:
        n1, n2, n3 = set_indice
        t_12 = G[n1][n2]['timestamp']
        t_13 = G[n1][n3]['timestamp']
        t_23 = G[n2][n3]['timestamp']
        times = [G[edge[0]][edge[1]]['timestamp'] for edge in combinations(set_indice, 2)]

        permutations = np.array( [
            [t_12, t_13, t_23],
            [t_12, t_23, t_13],
            [t_13, t_12, t_23],
            [t_13, t_23, t_12],
            [t_23, t_12, t_13],
            [t_23, t_13, t_12],
         ] )
        times = np.array(sorted(times))
        index = np.argmax(np.all(permutations == times, axis=1) ) 
        stat_array[index] += 1
    
    print(stat_array)


def number_of_disconnected_nodes(G, set_indices):
    num = 0
    for set_indice in set_indices:
        for n in set_indice:
            if G.degree(n) == 2:
                num += 1
                break
    return num
    
def class_acc(labels, correct_predictions):
    class_acc = []
    for class_i in range(6):
        labels_cpu = labels.cpu().numpy()
        ind = labels_cpu == class_i
        correct_predictions_cpu = correct_predictions.cpu().numpy()
        acc = correct_predictions_cpu[ind].sum()/ind.sum()
        class_acc.append(acc)
    return class_acc

def rank_acc(predictions, labels, rank=2):
    class_rank = torch.argsort(predictions, dim=-1, descending=True)
    labels = labels.reshape((-1, 1)).repeat(1, rank)
    correct_predictions = torch.max(class_rank[:, :rank] == labels, dim=1)[0]

    acc = correct_predictions.sum()/labels.shape[0]

    return acc.cpu().item()

