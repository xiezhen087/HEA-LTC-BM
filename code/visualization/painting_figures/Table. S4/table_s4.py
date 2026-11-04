# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:36:39 2026

@author: Realunknown
"""
import os
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
import numpy as np
import shutil
import pandas as pd

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

def calculate_delta(structure):
    delta = 0
    ratio = []
    radius = []
    elements = [atom.specie.symbol for atom in structure]
    element_type = list(set(elements))
    try:
        for element in element_type:
            ratio.append((elements.count(element))/len(elements))
            radius.append(Element(element).atomic_radius_calculated)
        radius_ave = sum([ratio[i]*radius[i] for i in range(len(ratio))])
        delta = np.sqrt(sum([ratio[i]*(1-(radius[i]/radius_ave))**2 for i in range(len(radius))]))
    except:
        #print(element_type)
        delta = 1000
    return delta

num_valence_electrons =  {'Ac': 3, 'Ag': 11, 'Al': 3, 'Am': 9, 'Ar': 8, 'As': 5, 'At': 7, 'Au': 11, 'B': 3, 'Ba': 2, 'Be': 2, 'Bi': 5, 'Br': 7, 'C': 4, 'Ca': 2, 'Cd': 12, 'Ce': 4, 'Cl': 7, 'Cm': 10, 'Co': 9, 'Cr': 6, 'Cs': 1, 'Cu': 11, 'Dy': 12, 'Er': 14, 'Eu': 9, 'F': 7, 'Fe': 8, 'Fr': 1, 'Ga': 3, 'Gd': 10, 'Ge': 4, 'H': 1, 'He': 2, 'Hf': 4, 'Hg': 12, 'Ho': 13, 'I': 7, 'In': 3, 'Ir': 9, 'K': 1, 'Kr': 8, 'La': 3, 'Li': 1, 'Lu': 3, 'Mg': 2, 'Mn': 7, 'Mo': 6, 'N': 5, 'Na': 1, 'Nb': 5, 'Nd': 6, 'Ne': 8, 'Ni': 10, 'Np': 7, 'O': 6, 'Os': 8, 'P': 5, 'Pa': 5, 'Pb': 4, 'Pd': 10, 'Pm': 7, 'Po': 6, 'Pr': 5, 'Pt': 10, 'Pu': 8, 'Ra': 2, 'Rb': 1, 'Re': 7, 'Rh': 9, 'Rn': 8, 'Ru': 8, 'S': 6, 'Sb': 5, 'Sc': 3, 'Se': 6, 'Si': 4, 'Sm': 8, 'Sn': 4, 'Sr': 2, 'Ta': 5, 'Tb': 11, 'Tc': 7, 'Te': 6, 'Th': 4, 'Ti': 4, 'Tl': 3, 'Tm': 15, 'U': 6, 'V': 5, 'W': 6, 'Xe': 8, 'Y': 3, 'Yb': 16, 'Zn': 12, 'Zr': 4}

def calculate_vec(structure):
    vec = 0
    ratio = []
    vec_per_element = []
    elements = [atom.specie.symbol for atom in structure]
    element_type = list(set(elements))
    for element in element_type:
        ratio.append((elements.count(element))/len(elements))
        vec_per_element.append(num_valence_electrons[element])
    vec = sum([ratio[i]*vec_per_element[i] for i in range(len(ratio))])
    vec_per_element
    return vec,vec_per_element,element_type

if __name__ == '__main__' :
    #phase = "fcc_2"
    phase = "bcc"
    #save_path_name_list = ["Fe_single","Fe_noble","Fe_mix","Zr_total","Zr_single","Zr_noble","Zr_rare","Zr_mix","Cu_total","Cu_single","Cu_mix","Mg_total","Mg_single","Mg_mix","Ti_total","Ti_single","Ti_noble","Ti_rare","Ti_mix","Ti_mix_2"]
    #save_path_name_list = ["Fe_total","Fe_single","Fe_noble","Fe_mix","Zr_total","Zr_single","Zr_noble","Zr_rare","Zr_mix","Cu_total","Cu_single","Cu_mix","Mg_total","Mg_single","Mg_mix","Ti_total","Ti_single","Ti_noble","Ti_rare","Ti_rare_2","Ti_mix"]
    save_path_name_list = ["Zr_rare"]
    system_dict = {}
    for save_path_name in save_path_name_list:
        print(save_path_name)
        if phase == "bcc":
            file_path = "./data/250522_nsga_quinary_bcc_multi_goal_" + save_path_name + "/" #BCC  # TODO: set your local data path
        elif phase == "fcc_1":
            file_path = "./data/250327_nsga_quinary_fcc_multi_goal_" + save_path_name + "/" #FCC  # TODO: set your local data path
        elif phase == "fcc_2":
            file_path = "./data/250327_nsga_quinary_fcc_multi_goal_" + save_path_name + "/" #FCC  # TODO: set your local data path
        
        system = []
        filtered_system = []
        file_name_list = os.listdir(file_path)
        for file_index in file_name_list:
            file_index_list = os.listdir(file_path + file_index + "/")
            poscar_name_list = [ele for ele in file_index_list if ele[0] == "P"]
            for poscar_name in poscar_name_list:
                information = poscar_name.split("_")
                BM = float(information[-2])
                LTC = float(information[-1])
                if phase == "bcc":
                    #if BM < 29.47 or LTC < 18.01: #for figure
                    if BM < 20 or LTC < 10: #for table
                        continue
                elif phase == "fcc1" or "fcc_2":
                    #if BM < 52.24 or LTC < 35.174: #for figure
                    if BM < 20 or LTC < 10: #for table
                        continue
                poscar = Poscar.from_file(file_path + file_index + "/" + poscar_name,'r')
                structure = poscar.structure
                elements = structure.elements
                elements = [str(ele) for ele in elements]
                elements = list(set(elements))
                element = elements[0] + "-" + elements[1] + "-" + elements[2] + "-" + elements[3] + "-" + elements[4]
                system.append(element)
                
                #filtering
                vec, _, _ = calculate_vec(structure)
                delta = calculate_delta(structure)
                H_mix = calculate_delta_H_mix(structure)
                if (delta < 0.05) and (H_mix > -22 and H_mix < 5):
                    if (phase == "fcc_1" or phase == "fcc_2") and vec > 8.0:
                        filtered_system.append(element)
                    elif phase == "bcc" and vec < 6.8:
                        filtered_system.append(element)
        
        system = list(set(system))
        filtered_system = list(set(filtered_system))
        system_dict[save_path_name] = [system, filtered_system]
        data_rows = []

        for key, double_sublist in system_dict.items():
            sublist1, sublist2 = double_sublist
            data_rows.append([key, sublist1, sublist2])
        df = pd.DataFrame(data_rows, columns=["Group", "Systems", "Filtered Systems"])
        #df.to_csv( "./" + phase + "_systems_per_group_delta_0-05.csv", index=False, encoding="utf-8")
        