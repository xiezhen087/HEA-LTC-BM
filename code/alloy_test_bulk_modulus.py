# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:58:01 2024

@author: Realunknown
"""

import torch
from pymatgen.io.vasp import Poscar
from mixmodel import MixModel
from torch_geometric.loader import DataLoader
import sys
sys.path.append("./")
from data_combine import output_data
import random
from pymatgen.core.periodic_table import Element
import os

if __name__ == '__main__' :
    model_time = "save_time_1"
    elements_type = ["bi_alloy","ter_alloy","quater_alloy","qui_alloy"]
    #elements_type = ["ter_alloy"]
    structure_class = ["fcc","bcc","hcp"]
    #structure_class = ["fcc"]
    model_path = "../pre_trained_models/bulk_modulus/result/0625_loop_" + model_time + "_r_square_0.8799/"
    material_property = "bulk_modulus"
    y_list_total = []
    y_data = []
    for element_type in elements_type:
        y_list = []
        y_list.append(element_type)
        for sclass in structure_class:
            y_list.append(sclass)
            if sclass == "fcc":
                file_path = "../searching/crystal_structure_bulk_modulus/" + element_type + "/fcc/"
                file_names = []
                for root, dirs, files in os.walk(file_path):
                    for Dir in dirs:
                        file_names.append(os.path.join(root, Dir))
            elif sclass == "bcc":
                file_path = "../searching/crystal_structure_bulk_modulus/" + element_type + "/bcc/"
                file_names = []
                for root, dirs, files in os.walk(file_path):
                    for Dir in dirs:
                        file_names.append(os.path.join(root, Dir))
            elif sclass == "hcp":
                file_path = "../searching/crystal_structure_bulk_modulus/" + element_type + "/hcp/"
                file_names = []
                for root, dirs, files in os.walk(file_path):
                    for Dir in dirs:
                        file_names.append(os.path.join(root, Dir))
            for file_name in file_names:
                y_list_data = []
                y_list_data = y_list_data + y_list + [file_name.split("/")[-1]]
                poscar = Poscar.from_file(file_name + "/POSCAR" ,'r')
                structure = poscar.structure
                data = output_data(structure,material_property)
                model = MixModel()
                model.load_state_dict(torch.load(model_path + "model_mix_best_" + material_property + "_0625_except_pure_metal_loop_" + model_time + ".pth" ,map_location=torch.device('cpu')))
                #model.load_state_dict(torch.load(model_path + "model_bo_temp-477-0.230.pt" ,map_location=torch.device('cpu')))
                normalization = torch.load(model_path + "normalization.pt")
                model.eval()
        
                val_loader = DataLoader([data], batch_size= 1,shuffle= False)
                for batch in val_loader:
                    with torch.no_grad():
                        energy_per_atom = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
                y = float(energy_per_atom[0][0])*float(normalization[1]) + float(normalization[0])
                y_list_data = y_list_data + [y]
                y_list_total.append(y_list_data)
                y_data.append(y)