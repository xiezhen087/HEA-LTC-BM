# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 19:59:59 2025

@author: Realunknown
"""
import os
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
import numpy as np
import shutil

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
        print(element_type)
    return delta
    

if __name__ == '__main__' :
    #save_path_name_list = ["Fe_total","Fe_single","Fe_noble","Fe_mix","Zr_total","Zr_single","Zr_noble","Zr_rare","Zr_mix","Cu_total","Cu_single","Cu_mix","Mg_total","Mg_single","Mg_mix","Ti_total","Ti_single","Ti_noble","Ti_rare","Ti_mix"]
    save_path_name_list = ["Fe_total","Fe_single","Fe_noble","Fe_mix","Zr_total","Zr_single","Zr_noble","Zr_rare","Zr_mix","Cu_total","Cu_single","Cu_mix","Mg_total","Mg_single","Mg_mix","Ti_total","Ti_single","Ti_noble","Ti_rare","Ti_rare_2","Ti_mix"]
    #save_path_name_list = ["Zr_total"]
    for save_path_name in save_path_name_list:
        print(save_path_name)
        file_path = "../searching/log/formal/250522_nsga_quinary_bcc_multi_goal/250522_nsga_quinary_bcc_multi_goal_" + save_path_name + "/"
        save_path = "../searching/log/final/multi_goal/temp_250522_nsga_quinary_bcc_multi_goal/250522_nsga_quinary_bcc_multi_goal_" + save_path_name + "/"
        os.mkdir(save_path)
        os.mkdir( save_path + "less_than_0.045")
        os.mkdir( save_path + "less_than_0.065")
        os.mkdir( save_path + "more_than_0.065")
        
        file_name_list = os.listdir(file_path)
        for file_index in file_name_list:
            file_index_list = os.listdir(file_path + file_index + "/")
            poscar_name_list = [ele for ele in file_index_list if ele[0] == "P"]
            for poscar_name in poscar_name_list:
                poscar = Poscar.from_file(file_path + file_index + "/" + poscar_name,'r')
                structure = poscar.structure
                delta = calculate_delta(structure)
                
                if (delta > 0) and (delta < 0.045):
                    shutil.copy(file_path + file_index + "/" + poscar_name, save_path + "less_than_0.045")
                elif (delta >= 0.045) and (delta < 0.065):
                    shutil.copy(file_path + file_index + "/" + poscar_name, save_path + "less_than_0.065")
                elif (delta >= 0.065):
                    shutil.copy(file_path + file_index + "/" + poscar_name, save_path + "more_than_0.065")