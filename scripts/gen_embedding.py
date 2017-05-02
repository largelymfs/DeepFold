#! /usr/bin/env python
#################################################################################
#     File Name           :     gen_embedding.py
#     Created By          :     yang
#     Creation Date       :     [2017-01-26 21:54]
#     Last Modified       :     [2017-01-26 23:01]
#     Description         :      
#################################################################################
import argparse
import os
import numpy as np

from network import DeepFold
from distance_matrix import get_distance_matrix
from network import DeepFold

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Create embeddings for a protein structure. Output a numpy embedding.')
    parser.add_argument('pdb_file', metavar='pdb_file', help='an input pdb file')
    parser.add_argument('output_file', metavar='output_file', help='an output numpy embedding')
    parser.add_argument('--model', metavar='model', default=os.path.join(os.path.dirname(__file__), './../models/deepfold.model'), help='the network model to load')
    args = parser.parse_args()

    distance_matrix = get_distance_matrix(args.pdb_file).astype("float32")
    model = DeepFold(max_length=distance_matrix.shape[0], projection_level=1)
    model.load_from_file(args.model)
    embedding = model.get_embedding(distance_matrix)

    np.save(args.output_file, embedding)
