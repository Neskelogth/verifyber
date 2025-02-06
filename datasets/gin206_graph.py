from __future__ import print_function
import os
import os.path
import torch
import numpy as np
import itertools


import nibabel as nib
from utils.data.selective_loader_numba import load_streamlines as load_streamlines_fast
from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset
from sklearn.neighbors import KDTree


class GIN206KnnDataset(gDataset):
    def __init__(self,
                 sub_file,
                 root_dir,
                 k=5,
                 same_size=False,
                 transform=None,
                 return_edges=False,
                 split_obj=False,
                 train=True,
                 labels_dir=None):

        """
        Args:
            root_dir (string): root directory of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        with open(sub_file) as f:
            subjects = [s.strip() for s in f.readlines()]
        self.subjects = subjects
        self.k = k
        self.transform = transform
        self.return_edges = return_edges
        self.train = train
        self.same_size = same_size
        self.labels_dir = labels_dir

        if train:
            split_obj=False
        if split_obj:
            self.remaining = [[] for _ in range(len(subjects))]
        self.split_obj = split_obj


    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        item = self.getitem(idx)
        return item


    def getitem(self, idx):

        sub = self.subjects[idx]
        sub_dir = os.path.join(self.root_dir, 'sub-%s' % sub)
        T_file = os.path.join(sub_dir, 'sub-%s_var-GIN_full_tract_gt20mm.trk' % (sub))
        T = nib.streamlines.load(T_file, lazy_load=True)

        label_sub_dir = os.path.join(self.root_dir.rsplit('/',1)[0], self.labels_dir, 'sub-%s' % sub)
        label_file = os.path.join(label_sub_dir, 'sub-%s_var-GIN_labels_gt20mm.npy' % (sub))
        gt = np.load(label_file)

        if self.split_obj:
            if len(self.remaining[idx]) == 0:
                self.remaining[idx] = set(np.arange(T.header['nb_streamlines']))

            sample = {'points': np.array(list(self.remaining[idx])), 'gt': gt[list(self.remaining[idx])]}

        else:
            sample = {'points': np.arange(T.header['nb_streamlines']), 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        if self.split_obj:
            self.remaining[idx] -= set(sample['points'])
            sample['obj_idxs'] = sample['points'].copy()
            sample['obj_full_size'] = T.header['nb_streamlines']

        sample['name'] = T_file.split('/')[-1].rsplit('.', 1)[0]
        sample['dir'] = sub_dir

        if self.same_size:
            streams, lengths = load_streamlines_fast(T_file,
                                                    sample['points'].tolist())
        else:
            streams, lengths = load_streamlines_fast(T_file,
                                                    sample['points'].tolist())

        sample['points'] = self.build_graph_sample(streams, lengths, torch.from_numpy(sample['gt']))

        return sample


    def build_anatomical_edges(self, slices, l):
        e1 = set(np.arange(0, l - 1)) - set(slices.numpy() - 1)
        e2 = set(np.arange(1, l)) - set(slices.numpy())
        return torch.tensor([list(e1) + list(e2), list(e2) + list(e1)], dtype=torch.long)


    def build_knn(self, point_cloud, k):
        kd_tree = KDTree(point_cloud)
        idxs = kd_tree.query(point_cloud, k=k, return_distance=False, dualtree=True)
        edge_1 = torch.arange(len(point_cloud)).repeat_interleave(5).tolist()
        edge_2 = list(itertools.chain.from_iterable(idxs))

        return torch.tensor([edge_1 + edge_2, edge_2 + edge_1], dtype=torch.long)


    def build_graph_sample(self, streams, lengths, gt):

        lengths = torch.from_numpy(lengths).long()
        batch_vec = torch.arange(len(lengths)).repeat_interleave(lengths)
        batch_slices = torch.cat([torch.tensor([0]), lengths.cumsum(dim=0)])
        slices = batch_slices[1:-1]
        streams = torch.from_numpy(streams)
        l = streams.shape[0]
        graph_sample = gData(x=streams, lengths=lengths, bvec=batch_vec, pos=streams)

        if self.return_edges:

            anatomical_edges = self.build_anatomical_edges(slices, l)
            anatomical_edge_labels = torch.tensor([0] * anatomical_edges.shape[1])
            knn_edges = self.build_knn(streams, self.k)
            knn_edges_labels = torch.tensor([1] * knn_edges.shape[1])

            edges = torch.cat((anatomical_edges, knn_edges), dim=1)
            edges_labels = torch.cat((anatomical_edge_labels, knn_edges_labels), dim=0)

            graph_sample['edge_index'] = edges
            num_edges = graph_sample.num_edges
            edge_attr = torch.ones(num_edges,1)
            graph_sample['edge_attr'] = edge_attr
            graph_sample['edge_type'] = edges_labels


        graph_sample['y'] = gt

        return graph_sample
