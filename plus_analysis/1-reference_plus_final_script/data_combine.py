# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:18:41 2024

@author: Realunknown
"""

import csv
import numpy as np
import os
import torch
import pandas as pd
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp import Poscar
import sys
sys.path.append("./")
from mlp_data_processing import get_mlp_data
from cgcnn_data_processing import get_cgcnn_data

mp_id = [0]
type_a = [2,3,10,11]
type_b = [8,9]
type_c = [1,4,5,6,7,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
embedding_a = [2,9]
embedding_b = [8]
embedding_c = [10,11]

def change_type(data_mlp):
    data_line = []
    for j in range(len(data_mlp)):
        if j in mp_id:
            data_line.append(data_mlp[j])
        if (j in type_a) or (j in type_b):
        #if j in mp_id:#Thermal conduct
        #    data_line.append(data_mlp[i][j])#Thermal conduct
        #if (j in type_a) or (j in type_b):#Thermal conduct
            data_mlp[j] = int(data_mlp[j])
            data_line.append(data_mlp[j])
        if j in type_c:
            data_line.append(data_mlp[j])
    return data_line

def package(data):
    data_fin_line = []
    mp_id_data = []
    embedding_a_data = []
    embedding_b_data = []
    embedding_c_data = []
    embedding_d_data = []
    continuous_data = []
    for j in range(len(data)):
        if j in mp_id:
            mp_id_data.append(data[j])
        elif j in embedding_a:
            embedding_a_data.append(data[j])
        elif j in embedding_b:
            embedding_b_data.append(data[j])
        elif j in embedding_c:
            embedding_c_data.append(data[j])
        #elif j in cont_e:#Thermal conduct
        elif j in type_c:
            continuous_data.append(data[j])
    data_fin_line.append(mp_id_data)
    data_fin_line.append(torch.LongTensor(embedding_a_data))
    data_fin_line.append(torch.LongTensor(embedding_b_data))
    data_fin_line.append(torch.LongTensor(embedding_c_data))
    data_fin_line.append(torch.FloatTensor(continuous_data))
    return data_fin_line

def mlp_data(data_mlp):
    data = change_type(data_mlp)
    data = package(data)
    return data

def cgcnn_data(data_cgcnn):
    data_cgcnn.y = torch.tensor([data_cgcnn.y])
    return data_cgcnn

def package_data(data_mlp,data_cgcnn):
    #Merge the GCN and MLP data parts
    data = []
    data.append(data_mlp)
    data.append(data_cgcnn)
    return data

def output_data(structure,material_property):
    data_cgcnn = get_cgcnn_data(structure)
    data_cgcnn = cgcnn_data(data_cgcnn)
    data_mlp = get_mlp_data(structure,material_property)
    data_mlp = mlp_data(data_mlp)
    data_combine = package_data(data_mlp,data_cgcnn)
    return data_combine

if __name__ == '__main__' :
    material_property = "bulk_modulus"
    data_path = "../initial_structure/"
    poscar = Poscar.from_file(data_path + "CONTCAR_SQS_fcc",'r')
    structure = poscar.structure
    #Preparing data_sets.
    data_combine = output_data(structure,material_property)