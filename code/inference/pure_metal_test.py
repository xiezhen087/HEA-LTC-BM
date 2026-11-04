# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:58:01 2024

@author: Realunknown
"""

import optuna
import torch
from pymatgen.io.vasp import Poscar
import sys
import pathlib
_CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _CODE_DIR.parent
for _p in (_CODE_DIR / "models", _CODE_DIR / "data_processing"):
    sys.path.insert(0, str(_p))
from torch_geometric.loader import DataLoader
from mixmodel import MixModel
from data_combine import output_data
import random
from pymatgen.core.periodic_table import Element

fcc_metal_elements = ["Al","Ni","Cu","Pt","Pd","Au","Sn","Co"]
bcc_metal_elements = ["Fe","Cr","Mo","Nb","W","V"]
hcp_metal_elements = ["Ti","Nd","Zr","Ag"]

if __name__ == '__main__' :
    model_time = "save_time_4"
    structure_class = ["fcc","bcc","hcp"]
    # shipped pre-trained model (pre_trained_models/ in this repository)
    model_path = os.path.join(_REPO_ROOT, "pre_trained_models", "bulk_modulus", "result", "0625_loop_save_time_1_r_square_0.8799") + os.sep
    material_property = "bulk_modulus"
    y_list = []
    for sclass in structure_class:
        if sclass == "fcc":
            file_path = "./data/pure_metal_poscar/fcc/"  # TODO: set your local data path
            metal_elements = fcc_metal_elements
        elif sclass == "bcc":
            file_path = "./data/pure_metal_poscar/bcc/"  # TODO: set your local data path
            metal_elements = bcc_metal_elements
        elif sclass == "hcp":
            file_path = "./data/pure_metal_poscar/hcp/"  # TODO: set your local data path
            metal_elements = hcp_metal_elements
        for element in metal_elements:
            poscar = Poscar.from_file(file_path + "POSCAR_" + element ,'r')
            structure = poscar.structure
            data = output_data(structure,material_property)
            model = MixModel()
            model.load_state_dict(torch.load(model_path + "model_mix_best_bulk_modulus_0625_except_pure_metal_loop_save_time_1_simple.pth" ,map_location=torch.device('cpu')))
            #model.load_state_dict(torch.load(model_path + "model_bo_temp-477-0.230.pt" ,map_location=torch.device('cpu')))
            normalization = torch.load(model_path + "normalization.pt")
            model.eval()

            val_loader = DataLoader([data], batch_size= 1,shuffle= False)
            for batch in val_loader:
                with torch.no_grad():
                    energy_per_atom = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
            y = float(energy_per_atom[0][0])*float(normalization[1]) + float(normalization[0])
            y_list.append(y)