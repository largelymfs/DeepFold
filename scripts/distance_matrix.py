#! /usr/bin/env python
#################################################################################
#     File Name           :     distance_matrix.py
#     Created By          :     Qing Ye
#     Creation Date       :     [2016-08-31 17:10]
#     Last Modified       :     [2017-01-26 22:10]
#     Description         :     generating distance matrix
#################################################################################
import sys
import os
import numpy
import scipy
import scipy.spatial

from Bio.PDB import PDBParser


def get_distance_matrix(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_path).get_list()[0]
    residue_positions = get_residue_positions(structure)
    pdb_dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(residue_positions, 'euclidean'))
    pdb_dist_mat[numpy.isnan(pdb_dist_mat)] = float('inf')
    return pdb_dist_mat


def get_residue_ids(structure):
    ids = [r.get_id()[1] for r in structure.get_residues()]
    return ids


def get_residue_positions(structure):
    residue_ids = get_residue_ids(structure)
    positions = numpy.ones((residue_ids[-1] - residue_ids[0] + 1, 3)) * float('inf')
    for residue in structure.get_residues():
        atoms = residue.get_atom()
        for a in atoms:
            if a.get_name() == 'CA':
                positions[residue.get_id()[1] - residue_ids[0]] = a.get_coord()
    return positions


if __name__ == '__main__':
    mat = get_distance_matrix('../examples/d2c5lc1.pdb')

