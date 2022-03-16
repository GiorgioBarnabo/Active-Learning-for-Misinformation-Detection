import time
import numpy as np
import os
import random

datasets = ['condor', 'gossipcop', 'politifact']

for dataset in datasets:

    graph_labels = np.load('{}/raw/graph_labels.npy'.format(dataset))

    train_idx = []
    val_idx = []
    test_idx = []

    for graph_index in range(len(graph_labels)):
        rand = random.random()
        if rand <= 0.01:
            train_idx.append(graph_index)
        elif rand <= 0.3:
            val_idx.append(graph_index)
        else:
            test_idx.append(graph_index)

    np.save('{}/raw/test_idx.npy'.format(dataset), np.array(test_idx))
    np.save('{}/raw/val_idx.npy'.format(dataset), np.array(val_idx))
    np.save('{}/raw/train_idx.npy'.format(dataset), np.array(train_idx))