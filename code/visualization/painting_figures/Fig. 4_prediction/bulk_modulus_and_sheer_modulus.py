# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 15:44:54 2025

@author: Realunknown
"""

import pandas as pd
import torch
import os

if __name__ == '__main__' :
    old_data_path = "./data/bulk_modulus/"  # TODO: set your local data path
    mlp_path = old_data_path + "features_normalized_bulk_modulus_log.csv"
    cgcnn_path = old_data_path + "cgcnn-voronoi-bulk_modulus-HEA.pt"
    raw_data_path = "./data/elasticity_1w/"  # TODO: set your local data path
    save_path = "./data/shear_modulus/"  # TODO: set your local data path
    mlp_data = pd.read_csv(mlp_path)
    cgcnn_data = torch.load(cgcnn_path)
    i = 0
    for i in range(len(mlp_data)):
        print(i)
        mp_id = mlp_data["material_id"][i]
        shear_modulus = torch.load(raw_data_path + "/" + mp_id + "/" + mp_id + "-20230409-elasticity.pt" )[1]["G_VRH"]
        mlp_data["target"][i] = shear_modulus
        cgcnn_data[i].y = shear_modulus
    mlp_data.to_csv(save_path + "features_normalized_shear_modulus_log.csv", index=False)
    torch.save(cgcnn_data, save_path + "cgcnn-voronoi-shear_modulus-HEA.pt")