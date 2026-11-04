# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:17:08 2024

@author: Realunknown
"""
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data_path = "./elasiticity/"
#data_path = "./thermal_conduct/"
header = ['density','space_group' ,'lattice_type','volume_pre_atom','volume_pre_cell','order_a_mean','order_a_variance','order_b_mean','order_b_variance','order_g','order_o','variance_of_atom_radius','extreme_atom_radius_mean_z_score','number_of_species','number_of_atom_per_cell','is_gas','is_metal','metal_elements_proportion','magnetic_elements_proportion','fw_atomic_number','fw_valence_electrons','fw_atomic_mass','fw_group','fw_period','fw_electronegativity','fw_boiling_point','fw_melting_point','fw_van_der_Waals_radii','fw_atomic_radius_calculated','fw_electrical_resistivity','fw_velocity_of_sound','fw_thermal_conductivity','fw_bulk_modulus','fw_youngs_modulus','fw_coefficient_of_linear_thermal_expansion','fw_ionization_energies']
maxmin_index = [1,4,5,6,7,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

if __name__ == '__main__' :
    #data_mlp = torch.load(data_path + "data_for_mlp.pt")
    #df_ini = pd.DataFrame(data_mlp, columns = header)
    #df = pd.DataFrame(data_mlp, columns = header)
    df = pd.read_csv(data_path + 'features_bulk_modulus_with_label.csv')
    #df = pd.read_csv(data_path_thermal_conduct + 'characters_phonon_thermal_conduct_no_index.csv')
    #df["density"] = df["density"].apply(np.log)
    df["volume_pre_atom"] = df["volume_pre_atom"].apply(np.log10)
    df["volume_pre_cell"] = df["volume_pre_cell"].apply(np.log10)
    df["variance_of_atom_radius"] = df["variance_of_atom_radius"].apply(np.log)
    #df["extreme_atom_radius_mean_z_score"] = df["extreme_atom_radius_mean_z_score"].apply(np.log)
    df["fw_electrical_resistivity"] = df["fw_electrical_resistivity"].apply(np.log10)
    df["fw_velocity_of_sound"] = df["fw_velocity_of_sound"].apply(np.log10)
    df["fw_thermal_conductivity"] = df["fw_thermal_conductivity"].apply(np.log10)
    df["fw_bulk_modulus"] = df["fw_bulk_modulus"].apply(np.log10)
    df["fw_youngs_modulus"] = df["fw_youngs_modulus"].apply(np.log10)
    df["fw_coefficient_of_linear_thermal_expansion"] = df["fw_coefficient_of_linear_thermal_expansion"].apply(np.log10)
    #df_new = df.replace(to_replace = -np.inf,value = 0)
    df_fin = df.replace(to_replace = -np.inf,value = 0)
    data_min_list = []
    data_max_list = []
    for i in maxmin_index:
        data_min = df_fin.iloc[:,i].min()
        data_min_list.append(data_min)
        data_max = df_fin.iloc[:,i].max()
        data_max_list.append(data_max)
        df_fin.iloc[:,i] = (df_fin.iloc[:,i] - data_min)/(data_max - data_min)
#    df_fin.to_csv(data_path + 'features_normalized_bulk_modulus_log.csv')