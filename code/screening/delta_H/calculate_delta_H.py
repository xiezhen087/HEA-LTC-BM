# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 13:50:02 2025

@author: Realunknown

All data in this script are from the paper "Classification of Bulk Metallic Glasses by Atomic Size Difference,
Heat of Mixing and Period of Constituent Elements and Its Application to Characterization of the Main Alloying Element"
DOI https://doi.org/10.2320/matertrans.46.2817 

"""
import pandas as pd
import os
from pymatgen.io.vasp import Poscar
from pymatgen.core.periodic_table import Element

delta_H_a_path = "./ΔHmin_a.csv"
delta_H_b_up_path = "./ΔHmin_b_up.csv"
delta_H_b_down_path = "./ΔHmin_b_down.csv"
delta_H_a = pd.read_csv(delta_H_a_path, header = 0,index_col = 0)
delta_H_b_up = pd.read_csv(delta_H_b_up_path, header = 0,index_col = 0, encoding='gbk')
delta_H_b_down = pd.read_csv(delta_H_b_down_path, header = 0,index_col = 0)
delta_H_a_number_row = [1,3,4,5,6,7,11,12,13,14,15,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,37,38,39,40,41,42,43,44,45,46]
delta_H_a_number_column = [46,47,48,49,50,51,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,90,92,94]
delta_H_b_number_up = [1,3,4,5,6,7,11,12,13,14,15,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,37,38,39,40,41,42,43,44,45]
delta_H_b_number_down = [47,48,49,50,51,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,90,92,94]

def get_delta_H_mix_AB(Z_i,Z_j):
    Z_max = max(Z_i,Z_j)
    Z_min = min(Z_i,Z_j)
    if Z_min in delta_H_a_number_row and Z_max in delta_H_a_number_column:
        delta_H_mix_AB = delta_H_a[str(Z_min)][Z_max]
    elif Z_min in  delta_H_b_number_up and Z_max in delta_H_b_number_up:
        delta_H_mix_AB = delta_H_b_up[str(Z_max)][Z_min]
    elif Z_min in  delta_H_b_number_down and Z_max in delta_H_b_number_down:
        delta_H_mix_AB = delta_H_b_down[str(Z_min)][Z_max]
    return delta_H_mix_AB

def calculate_delta_H_mix(structure):
    elements = [atom.specie.Z for atom in structure]
    element = list(set(elements))
    elements_ratio = [elements.count(ele)/(len(elements)) for ele in element]
    H_mix_ij_list = []
    for i in range(len(element) - 1):
        for j in range(i + 1,len(element)):
            Z_i = element[i]
            Z_j = element[j]
            c_i = elements_ratio[i]
            c_j = elements_ratio[j]
            delta_H_mix_AB = get_delta_H_mix_AB(Z_i,Z_j)
            delta_H_mix_AB = float(delta_H_mix_AB)
            """
            print(Element.from_Z(element[i]))
            print(Element.from_Z(element[j]))
            print(delta_H_mix_AB)
            print("***************************")
            """
            H_mix_ij = 4 * delta_H_mix_AB * c_i * c_j
            H_mix_ij_list.append(H_mix_ij)
    H_mix = sum(H_mix_ij_list)
    return H_mix

if __name__ == '__main__' :
    pro = "multi_object"
    structure_path = "./data/final_used_structure/" + pro + "/"  # TODO: set your local data path
    structure_files = os.listdir(structure_path)
    for file in structure_files:
        poscar = Poscar.from_file(structure_path + file)
        structure = poscar.structure
        H_mix = calculate_delta_H_mix(structure)
        #if H_mix => -15 and H_mix <= 5:
        #    pass
        #else:
        print(H_mix)
        print(file)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #break
    """
    structure_path = "./data/efficient_data_non_equiatoms/"
    structure_files = os.listdir(structure_path)
    for file in structure_files:
        poscar = Poscar.from_file(structure_path + file)
        structure = poscar.structure
        H_mix = calculate_delta_H_mix(structure)
        #if H_mix => -15 and H_mix <= 5:
        #    pass
        #else:
        print(H_mix)
        print(file)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #break
    """