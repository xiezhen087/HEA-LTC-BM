# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:58:01 2024

@author: Realunknown
"""

import optuna
import torch
from pymatgen.io.vasp import Poscar
from mixmodel import MixModel
from torch_geometric.loader import DataLoader
import sys
sys.path.append("./")
from data_combine import output_data
import random
from pymatgen.core.periodic_table import Element

fcc_metal_elements = ["Al","Ni","Cu","Pt","Pd","Au","Sn","Co"]
bcc_metal_elements = ["Fe","Cr","Mo","Nb","W","V"]
hcp_metal_elements = ["Ti","Nd","Zr","Ag"]

if __name__ == '__main__' :
    model_time = "save_time_4"
    structure_class = ["fcc","bcc","hcp"]
    model_path = "../pre_trained_models/bulk_modulus/result/0624_loop_" + model_time + "/"
    material_property = "bulk_modulus"
    y_list = []
    for sclass in structure_class:
        if sclass == "fcc":
            file_path = "../searching/pure_metal_poscar/fcc/"
            metal_elements = fcc_metal_elements
        elif sclass == "bcc":
            file_path = "../searching/pure_metal_poscar/bcc/"
            metal_elements = bcc_metal_elements
        elif sclass == "hcp":
            file_path = "../searching/pure_metal_poscar/hcp/"
            metal_elements = hcp_metal_elements
        for element in metal_elements:
            poscar = Poscar.from_file(file_path + "POSCAR_" + element ,'r')
            structure = poscar.structure
            data = output_data(structure,material_property)
            model = MixModel()
            model.load_state_dict(torch.load(model_path + "model_mix_best_" + material_property + "_0624_except_pure_metal_loop_" + model_time + ".pth" ,map_location=torch.device('cpu')))
            #model.load_state_dict(torch.load(model_path + "model_bo_temp-477-0.230.pt" ,map_location=torch.device('cpu')))
            normalization = torch.load(model_path + "normalization.pt")
            model.eval()

            val_loader = DataLoader([data], batch_size= 1,shuffle= False)
            for batch in val_loader:
                with torch.no_grad():
                    energy_per_atom = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
            y = float(energy_per_atom[0][0])*float(normalization[1]) + float(normalization[0])
            y_list.append(y)