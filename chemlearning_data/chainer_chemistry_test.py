#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, E. Nicolas

"""Tools to use data (especially from QM9) for machine learning applications."""

# Here comes your imports
import cclib
import tarfile
import os
import chainer
from chainer.datasets import split_dataset_random
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.models import MLP, NFP

# Here comes your (few) global variables

# Here comes your class definitions
class GraphConvPredictor(chainer.Chain):

    def __init__(self, graph_conv, mlp):
        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            self.mlp = mlp

    def __call__(self, atoms, adjs):
        x = self.graph_conv(atoms, adjs)
        x = self.mlp(x)
        return x


# Here comes your function definitions


def main():
    """Launcher."""
    preprocessor = preprocess_method_dict["nfp"]()
    dataset = datasets.get_qm9(preprocessor, labels="homo")
    cache_dir = "data/"
    if not(os.path.exists(cache_dir)):
        os.makedirs(cache_dir)
    NumpyTupleDataset.save(cache_dir + "data.npz", dataset)
    dataset = NumpyTupleDataset.load(cache_dir + 'data.npz')
    train_data_ratio = 0.7
    train_data_size = int(len(dataset) * train_data_ratio)
    train, validation = split_dataset_random(dataset, train_data_size, 777)
    print('train dataset size:', len(train))
    print('validation dataset size:', len(validation))

    n_unit = 16
    conv_layers = 4
    model = GraphConvPredictor(NFP(n_unit, n_unit, conv_layers),
                               MLP(n_unit, 1))


if __name__ == "__main__":
    main()
