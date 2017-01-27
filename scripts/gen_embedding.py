#! /usr/bin/env python
#################################################################################
#     File Name           :     gen_embedding.py
#     Created By          :     yang
#     Creation Date       :     [2017-01-26 21:54]
#     Last Modified       :     [2017-01-26 22:37]
#     Description         :      
#################################################################################
from network import DeepFold
from distance_matrix import get_distance_matrix
import numpy as np
from network import DeepFold

if __name__=="__main__":
    distance_matrix = get_distance_matrix("./../examples/d2c5lc1.pdb").astype("float32")
    max_length = 256
    model_file_name = "./../models/deepfold.model"
    model = DeepFold(max_length = max_length, projection_level = 1)
    model.load_from_file(model_file_name)
    print model.get_embedding(distance_matrix)


    

