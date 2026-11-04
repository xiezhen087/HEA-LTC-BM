# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 20:49:53 2025

@author: Realunknown
"""

import os
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
import numpy as np
import shutil

num_valence_electrons =  {'Ac': 3, 'Ag': 11, 'Al': 3, 'Am': 9, 'Ar': 8, 'As': 5, 'At': 7, 'Au': 11, 'B': 3, 'Ba': 2, 'Be': 2, 'Bi': 5, 'Br': 7, 'C': 4, 'Ca': 2, 'Cd': 12, 'Ce': 4, 'Cl': 7, 'Cm': 10, 'Co': 9, 'Cr': 6, 'Cs': 1, 'Cu': 11, 'Dy': 12, 'Er': 14, 'Eu': 9, 'F': 7, 'Fe': 8, 'Fr': 1, 'Ga': 3, 'Gd': 10, 'Ge': 4, 'H': 1, 'He': 2, 'Hf': 4, 'Hg': 12, 'Ho': 13, 'I': 7, 'In': 3, 'Ir': 9, 'K': 1, 'Kr': 8, 'La': 3, 'Li': 1, 'Lu': 3, 'Mg': 2, 'Mn': 7, 'Mo': 6, 'N': 5, 'Na': 1, 'Nb': 5, 'Nd': 6, 'Ne': 8, 'Ni': 10, 'Np': 7, 'O': 6, 'Os': 8, 'P': 5, 'Pa': 5, 'Pb': 4, 'Pd': 10, 'Pm': 7, 'Po': 6, 'Pr': 5, 'Pt': 10, 'Pu': 8, 'Ra': 2, 'Rb': 1, 'Re': 7, 'Rh': 9, 'Rn': 8, 'Ru': 8, 'S': 6, 'Sb': 5, 'Sc': 3, 'Se': 6, 'Si': 4, 'Sm': 8, 'Sn': 4, 'Sr': 2, 'Ta': 5, 'Tb': 11, 'Tc': 7, 'Te': 6, 'Th': 4, 'Ti': 4, 'Tl': 3, 'Tm': 15, 'U': 6, 'V': 5, 'W': 6, 'Xe': 8, 'Y': 3, 'Yb': 16, 'Zn': 12, 'Zr': 4}

def calculate_vec(structure):
    vec = 0
    ratio = []
    radius = []
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
    #save_path_name_list = ["Fe_total","Fe_single","Fe_noble","Fe_mix","Zr_total","Zr_single","Zr_noble","Zr_rare","Zr_mix","Cu_total","Cu_single","Cu_mix","Mg_total","Mg_single","Mg_mix","Ti_total","Ti_single","Ti_noble","Ti_rare","Ti_mix"]
    #save_path_name_list = ["Fe_single","Fe_noble","Fe_mix","Zr_total","Zr_single","Zr_noble","Zr_rare","Zr_mix","Cu_total","Cu_single","Cu_mix","Mg_total","Mg_single","Mg_mix","Ti_total","Ti_single","Ti_noble","Ti_rare","Ti_mix"]
    #save_path_name_list = ["Zr_rare","Zr_mix","Cu_total","Cu_single","Cu_mix","Mg_total","Mg_single","Mg_mix","Ti_total","Ti_single","Ti_noble","Ti_rare","Ti_rare_2","Ti_mix"]
    #save_path_name = "Mg_mix"
    # TODO: set your local data path (optimization log folder)
    src_path = "./data/temp_250522_nsga_quinary_bcc_multi_goal"
    file_path_list = os.listdir(src_path)
    for file_path in file_path_list:
        src = src_path + "/" + file_path + "/less_than_0.045"
        save_path = "./data/final_250522_nsga_quinary_bcc_multi_goal/" + file_path
        os.mkdir(save_path)
        
        file_name_list = os.listdir(src)
        poscar_name_list = [ele for ele in file_name_list if ele[0] == "P"]
        for poscar_name in poscar_name_list:
            poscar = Poscar.from_file(src + "/" + poscar_name,'r')
            structure = poscar.structure
            vec,vec_per_element,element_type = calculate_vec(structure)
            """
            if any(x > 8.0 for x in vec_per_element): #fcc
            #if vec <= 6.8: #bcc
                print(poscar_name)
                print(element_type)
                print(vec_per_element)
            """
            pred_value = poscar_name.split("_")
            #if vec >= 8.0: #fcc
            if vec <= 6.8 and (float(pred_value[-1]) > 20 and float(pred_value[-2]) > 30): #bcc
                print(poscar_name)
                print(vec)
                shutil.copy(src + "/" + poscar_name, save_path)