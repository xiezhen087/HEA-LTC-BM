# -*- coding: utf-8 -*-
"""
Created on Tue May  6 10:51:15 2025

@author: Realunknown
"""
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from pymoo.indicators.hv import HV
from mixmodel import MixModel
from torch_geometric.loader import DataLoader
import sys
sys.path.append("./")
from data_combine import output_data
import torch
from pymatgen.core.periodic_table import Element
import itertools
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure, Lattice
import torch.nn as nn
import time

src_files_path = "../searching/log/formal/250327_nsga_quinary_fcc_multi_goal/"

class Individual(object):
    def __init__(self):
        self.solution = None
        self.objective = defaultdict()
 
        self.n = 0
        self.rank = 0
        self.S = []
        self.distance = 0
 
    def bound_process(self, bound_min, bound_max):
        for i, item in enumerate(self.solution):
            if item > bound_max[i]:
                self.solution[i] = bound_max[i]
            elif item < bound_min[i]:
                self.solution[i] = bound_min[i]
 
    def calculate_objective(self, objective_fun):
        self.objective = objective_fun(self.solution)
 
    def __lt__(self, other):
        v1 = list(self.objective.values())
        v2 = list(other.objective.values())
        for i in range(len(v1)):
            if v1[i] > v2[i]:#define the objective optimization direction
                return 0 
        return 1
    
def modify_lattice_constant(structure,framework):
    lattice_constant = [0]*3
    species = structure.species
    num_species = len(set(species))
    species_type = list(set(species))
    species_radius_type = [ele.atomic_radius_calculated for ele in species_type]
    species_ratio = [species.count(ele)/len(species) for ele in species_type]
    if (num_species == 2) and framework == "fcc":
        lattice_constant[0] = 0.7069*sum(species_radius_type)+1.5677
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = lattice_constant[0]
    elif (num_species == 2) and framework == "bcc":
        lattice_constant[0] = 1.0884*sum(np.array(species_radius_type)*np.array(species_ratio))+1.2315
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = lattice_constant[0]
    elif (num_species == 2) and framework == "hcp":
        lattice_constant[0] = 1.3335*sum(species_radius_type)+1.2001
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = 8.2611*sum(np.array(species_radius_type)*np.array(species_ratio))-6.0775
    elif (num_species == 3) and framework == "fcc":
        lattice_constant[0] = 3.64570
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = lattice_constant[0]
    elif (num_species == 5) and framework == "fcc":
        lattice_constant[0] = 3.844382553
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = lattice_constant[0]
    elif (num_species == 5) and framework == "bcc":
        lattice_constant[0] = 3.239816519
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = lattice_constant[0]
    lattice = structure.lattice
    new_lattice = Lattice.from_parameters( a=lattice_constant[0]*3, b=lattice_constant[1]*3, c=lattice_constant[2]*3, alpha=lattice.alpha, beta=lattice.beta, gamma=lattice.gamma)
    new_structure = Structure(new_lattice, structure.species, structure.frac_coords)
    return new_structure

def re_construct_structure(structure ,element_list, ratio_list, framework):
    total_atom_num = len(structure)
    element_num = [int(total_atom_num * ratio) for ratio in ratio_list]
    element_num.append(int(total_atom_num - sum(element_num)))
    element_num = [0] + element_num 
    
    index_list = [i for i in range(total_atom_num)]
    random.shuffle(index_list)
    element_index_part = []
    for i in range(len(element_num) -1):
        ini_index = sum(element_num[0:i+1])
        end_index = sum(element_num[0:i + 2])
        element_index_part.append(index_list[ini_index:end_index])
    
    for i in range(len(element_index_part)):
        for j in range(len(element_index_part[i])):
           structure.replace(element_index_part[i][j], Element(element_list[i]))
    return structure

def input_poscar(element_list, ratio_list, framework):
    file_path = "../initial_structure/"
    num_species = len(element_list)
    if num_species == 2:
        if framework == "fcc":
            poscar = Poscar.from_file(file_path + "bi_alloy/fcc/" + "POSCAR_Fe_fcc",'r')
        elif framework == "bcc":
            poscar = Poscar.from_file(file_path + "bi_alloy/bcc/" + "POSCAR_W_bcc",'r')
        elif framework == "hcp":
            poscar = Poscar.from_file(file_path + "bi_alloy/hcp/" + "POSCAR_Zr_hcp",'r')
    elif num_species == 3:
        if framework == "fcc":
            poscar = Poscar.from_file(file_path + "ter_alloy/fcc/" + "POSCAR_Fe_fcc",'r')
        elif framework == "bcc":
            poscar = Poscar.from_file(file_path + "ter_alloy/bcc/" + "POSCAR_W_bcc",'r')
        elif framework == "hcp":
            poscar = Poscar.from_file(file_path + "ter_alloy/hcp/" + "POSCAR_Zr_hcp",'r')
    elif num_species == 5:
        if framework == "fcc":
            poscar = Poscar.from_file(file_path + "qui_alloy/fcc/" + "POSCAR_Fe_fcc",'r')
        elif framework == "bcc":
            poscar = Poscar.from_file(file_path + "qui_alloy/bcc/" + "POSCAR_W_bcc",'r')
        elif framework == "hcp":
            poscar = Poscar.from_file(file_path + "qui_alloy/hcp/" + "POSCAR_Zr_hcp",'r')
    structure = poscar.structure
    supercell = structure.make_supercell([3, 3, 3])
    structure = Structure(supercell.lattice, supercell.species, supercell.frac_coords)
    structure = re_construct_structure(structure ,element_list, ratio_list, framework)
    structure = modify_lattice_constant(structure,framework)
    return structure

def load_data(structure, material_property):
    data = output_data(structure,material_property)
    return data

def individual_to_data(x_individual):
    element_num = len(x_individual) - 1
    ratio_list = [0]*(element_num - 1)
    
    element_list = all_combinations[int(x_individual[0])].split(" ")
    for i in range(element_num - 1):
        ratio_list[i] = float(x_individual[i + 1])
    #print(x_individual)
    framework = selected_framework[int(x_individual[-1])]
    structure = input_poscar(element_list, ratio_list, framework)
    
    data_bulk_modulus = load_data(structure,"bulk_modulus")
    data_thermal_conduct = load_data(structure,"thermal_conduct")
    data = (data_bulk_modulus,data_thermal_conduct)
    return data, structure

if __name__ == '__main__' :
    nsga_file_list = os.listdir(src_files_path)[:-5]
    #nsga_file_list = ["250327_nsga_quinary_fcc_multi_goal_Fe_total"]
    element_num = 5
    counts = 0
    
    for nsga_file_name in nsga_file_list:
        save_path_name =  nsga_file_name[35:]
        if save_path_name == "Fe_total":
            selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Ru","Co","Rh","Pd","Ni","Pt","Cu","Ag","Au","Zn","Ga","Sn","Sb","Bi","La"]#Fe_total
        elif save_path_name == "Fe_single":
            selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Sn","Sb","Bi"]#Fe_single
        elif save_path_name == "Fe_noble":
            selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Ru","Co","Rh","Pd","Ni","Pt","Cu","Ag","Au","Zn","Ga","Sn","Sb","Bi"]#Fe_noble
        elif save_path_name == "Fe_mix":
            selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Sn","Sb","Bi","Sc","Tc","Ir","Mo","W"]#Fe_mix
        elif save_path_name == "Zr_total":
            selected_element_list = ["Ca","Sr","Sc","Y","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","Ru","Os","Ir","Pt","Nd","Pm","Eu","Gd","Tb","Dy","Er","Tm","Yb","Lu"]#Zr_total
        elif save_path_name == "Zr_single":
            selected_element_list = ["Ca","Sr","Ti","Zr","Hf","Nb","Ta","Mo","W","Re"]#Zr_single
        elif save_path_name == "Zr_noble":
            selected_element_list = ["Ca","Sr","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","Ru","Os","Ir","Pt"]#Zr_noble
        elif save_path_name == "Zr_rare":
            selected_element_list = ["Ca","Sr","Sc","Y","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","Nd","Pm","Eu","Gd","Tb","Dy","Er","Tm","Yb","Lu"]#Zr_rare
        elif save_path_name == "Zr_mix":
            selected_element_list = ["Ca","Sr","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","V","Cr","La","Pr"]#Zr_mix
        elif save_path_name == "Cu_total":
            selected_element_list = ["Mg","Cr","Mn","Fe","Co","Ni","Cu","Ag","Au","Zn","Ga","Ge","Sn","Sb","Bi"]#Cu_total
        elif save_path_name == "Cu_single":
            selected_element_list = ["Mg","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","Sn","Sb","Bi"]#Cu_single
        elif save_path_name == "Cu_mix":
            selected_element_list = ["Mg","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","Sn","Sb","Bi","Li","Pd","Al","La"]#Cu_mix
        elif save_path_name == "Mg_total":
            selected_element_list = ["Mg","Cr","Fe","Co","Ni","Cu","Ag","Zn","Ga","Ge","Sn","Sb","Bi"]#Mg_total
        elif save_path_name == "Mg_single":
            selected_element_list = ["Mg","Cr","Fe","Co","Ni","Cu","Zn","Ga","Ge","Sn","Sb","Bi"]#Mg_single
        elif save_path_name == "Mg_mix":
            selected_element_list = ["Mg","Cr","Fe","Co","Ni","Cu","Ag","Zn","Ga","Ge","Sn","Sb","Bi","Li","V","Mn","La","Al"]#Mg_mix
        elif save_path_name == "Ti_total":
            selected_element_list = ["Li","Ca","Sc","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Ru","Os","Co","Rh","Pd","Ir","Pt","Ag","Au","La"]#Ti_total
        elif save_path_name == "Ti_single":
            selected_element_list = ["Li","Ca","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Co"]#Ti_single
        elif save_path_name == "Ti_noble":
            selected_element_list = ["Li","Ca","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Ru","Os","Co","Rh","Pd","Ir","Pt","Ag","Au"]#Ti_noble
        elif save_path_name == "Ti_rare":
            selected_element_list = ["Li","Ca","Sc","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Co","La"]#Ti_rare
        elif save_path_name == "Ti_mix" or save_path_name == "Ti_mix_2":
            selected_element_list = ["Li","Ca","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Co","Mg","Zr","Hf","Ni","Cu","Sn","Bi","Nd","Pm"]#Ti_mix
        
        all_combinations = list(itertools.combinations(selected_element_list, element_num))
        all_combinations = [" ".join(list(ele)) for ele in all_combinations]
        selected_framework = ["fcc"]
                
        nsga_file_sub_file_list = os.listdir(src_files_path + nsga_file_name + "/")
        os.mkdir(src_files_path + "structure_and_data_selected_ltc_below_60/" + nsga_file_name + "/" )
        for nsga_file_sub_file_name in nsga_file_sub_file_list:
            if (nsga_file_sub_file_name[-2:] == "pt") and ("x" not in nsga_file_sub_file_name):
                structure_list = torch.load(src_files_path + nsga_file_name + "/" + nsga_file_sub_file_name)
                data_list = torch.load(src_files_path + nsga_file_name + "/" + nsga_file_sub_file_name[:35] + "plot_x_y" + nsga_file_sub_file_name[-6:])
                os.mkdir(src_files_path + "structure_and_data_selected_ltc_below_60/" + nsga_file_name + "/" + nsga_file_sub_file_name[-5:-3])
                for i in range(len(data_list[0])):
                    #if data_list[0][i] > 40 and (data_list[1][i] > 35 and data_list[1][i] <= 60):
                    if data_list[0][i] > 20 and (data_list[1][i] > 10):
                        counts += 1
                        _,structure = individual_to_data(structure_list[i].solution)
                        elements = structure.species
                        coords = structure.frac_coords
                        sorted_indices = sorted(range(len(elements)), key=lambda k: elements[k])
                        sorted_elements = [elements[l] for l in sorted_indices]
                        sorted_coords = [coords[m] for m in sorted_indices]
                        new_structure = Structure(structure.lattice, sorted_elements, sorted_coords)
                        
                        new_structure.to(src_files_path + "structure_and_data_selected_ltc_below_60/" + nsga_file_name + "/" + nsga_file_sub_file_name[-5:-3] +"/" + "/POSCAR_" + str(int(structure_list[i].solution[0])) + '_{:.2f}'.format(data_list[0][i]) + '_{:.2f}'.format(data_list[1][i]),fmt = "poscar")
                        #break
            #break
    