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

pure_metal_id = ["mp-134","mp-23","mp-30","mp-126","mp-2","mp-81","mp-117","mp-102","mp-13","mp-90","mp-129","mp-75","mp-91","mp-146","mp-72","mp-123","mp-131","mp-8566"]

mp_id = [0]
type_a = [2,3,10,11]
type_b = [8,9]
type_c = [1,4,5,6,7,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
embedding_a = [2,9]
embedding_b = [8]
embedding_c = [10,11]

def change_type(data_mlp):
    data = []
    for i in range(len(data_mlp)):
        data_line = []
        for j in range(len(data_mlp[0])):
            if j in mp_id:
                data_line.append(data_mlp[i][j])
            if (j in type_a) or (j in type_b):
            #if j in mp_id:#Thermal conduct
            #    data_line.append(data_mlp[i][j])#Thermal conduct
            #if (j in type_a) or (j in type_b):#Thermal conduct
                data_mlp[i][j] = int(data_mlp[i][j])
                data_line.append(data_mlp[i][j])
            if j in type_c:
                data_line.append(data_mlp[i][j])
        data.append(data_line)
    return data

def package(data):
    data_fin = []
    for i in range(len(data)):
        data_fin_line = []
        mp_id_data = []
        embedding_a_data = []
        embedding_b_data = []
        embedding_c_data = []
        embedding_d_data = []
        continuous_data = []
        for j in range(len(data[i])):
            if j in mp_id:
                mp_id_data.append(data[i][j])
            elif j in embedding_a:
                embedding_a_data.append(data[i][j])
            elif j in embedding_b:
                embedding_b_data.append(data[i][j])
            elif j in embedding_c:
                embedding_c_data.append(data[i][j])
            #elif j in cont_e:#Thermal conduct
            elif j in type_c:
                continuous_data.append(data[i][j])
        data_fin_line.append(mp_id_data)
        #data_fin_line.append(mp_id_data)#Thermal conduct
        data_fin_line.append(torch.LongTensor(embedding_a_data))
        data_fin_line.append(torch.LongTensor(embedding_b_data))
        data_fin_line.append(torch.LongTensor(embedding_c_data))
        data_fin_line.append(torch.FloatTensor(continuous_data))
        data_fin.append(data_fin_line)
    return data_fin

def mlp_data():
    mlp_data_path = "./elasiticity/"
    data_raw = pd.read_csv(mlp_data_path + "features_normalized_bulk_modulus_log.csv")
    #mlp_data_path = "./thermal_conduct/"#Thermal conduct
    #data_raw = pd.read_csv(mlp_data_path + "features_thermal_conductivity_with_label.csv")#Thermal conduct
    data_mlp = data_raw.values.tolist()
    data = change_type(data_mlp)
    data = package(data)
    return data

def cgcnn_data():
    cgcnn_data_path = './elasiticity/'
    data_set = torch.load(os.path.join(cgcnn_data_path + 'cgcnn-voronoi-bulk_modulus-HEA.pt'))#Bulk Modulus
    #data_set = torch.load(os.path.join(cgcnn_data_path + '/Thermo_conduct/cgcnn-voronoi-Phonon_thermal_conduct.pt'))#Thermal conduct
    for i in range(len(data_set)):
        data_set[i].y = torch.tensor([data_set[i].y])
    return data_set

def package_data(data_mlp,data_cgcnn):
    #Merge the GCN and MLP data parts
    data = []
    DATA = []
    for i in range(len(data_mlp)):
        if len(data_cgcnn[i].edge_attr) != len(data_cgcnn[i].edge_index[0]):#one sample is missing two edge features
            continue
        DATA.append(data_mlp[i])
        DATA.append(data_cgcnn[i])
        data.append(DATA)
        DATA = []
    return data

if __name__ == '__main__' :
    save_path = "./elasiticity/"
    #save_path = "./thermal_conduct/"
    #Preparing data_sets.
    data_cgcnn = cgcnn_data()
    data_mlp = mlp_data()
    data_set = package_data(data_mlp, data_cgcnn)
    pure_metal_data_set = []
    for data in data_set:
        if data[0][0][0] not in pure_metal_id:
            pure_metal_data_set.append(data)
#    torch.save(pure_metal_data_set,save_path + "mix-voronoi-bulk_modulus-except_pure_metal-HEA.pt")
    #torch.save(data_set,save_path + "mix-voronoi-thermal_conduct-HEA.pt")