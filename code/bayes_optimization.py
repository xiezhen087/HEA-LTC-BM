# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:03:14 2024

@author: PC
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
import itertools
from pymatgen.core import Structure, Lattice
import numpy as np
import os

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
        lattice_constant[0] = 6.1366
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
    structure = modify_lattice_constant(structure,framework)
    return structure

def calculate_y(data, material_property):
    if  material_property == "bulk_modulus":
        model_path = "../pre_trained_models/bulk_modulus/result/0625_loop_save_time_1_r_square_0.8799/"
        model = MixModel()
        model.load_state_dict(torch.load(model_path + "model_mix_best_" + material_property + "_0625_except_pure_metal_loop_save_time_1.pth" ,map_location=torch.device('cpu')))
    elif  material_property == "thermal_conduct":
        model_path = "../pre_trained_models/thermal_conduct/result/0802_loop_save_time_0_r_square_0.872/"
        model = MixModel()
        model.load_state_dict(torch.load(model_path + "model_mix_best_" + material_property + "_0802_except_pure_metal_loop_save_time_0.pth" ,map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load(model_path + "model_bo_temp-477-0.230.pt" ,map_location=torch.device('cpu')))
    normalization = torch.load(model_path + "normalization.pt")
    model.eval()
    
    val_loader = DataLoader([data], batch_size= 1,shuffle= False)
    for batch in val_loader:
        with torch.no_grad():
            energy_per_atom = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
    print("Bulk Modulus:{}".format(energy_per_atom[0][0]))
    y = float(energy_per_atom[0][0])*float(normalization[1]) + float(normalization[0])
    print("y:{}".format(y))
    return y

def objective(trial):
    element_num = 4 #quaternary high entropy alloy
    material_property = "thermal_conduct"
    #material_property = "bulk_modulus"
    ratio_list = [0]*(element_num -1)
    
    elements = trial.suggest_categorical('elements', all_combinations)
    element_list = elements.split(" ")
    #fcc pure_metal = ["Al","Ni","Cu","Pt","Pd","Au","Rh","Th","Yb","Co"]
    #bcc pure_metal = ["Fe","Cr",Mo","Nb","W","V"]
    #hcp pure_metal = ["Ti","Nd","Zr","Ag"]
    #element_list = ["Yb","Yb","Yb","Yb"]
    
    # searching ratio
    #for i in range(element_num -1):
    #    ratio_list[i] = trial.suggest_float(f'ratio_{i}',0.05,0.2)
    ratio_list = [0.25,0.25,0.25]
    #ratio_list = [0.5]
        
    #searching framework
    framework =  trial.suggest_categorical('framwork', ["fcc"])
    #framework = "fcc"
    
    if len(study.trials) == 1:
        structure = input_poscar(element_list, ratio_list, framework)
    else:
        try:
            poscar = Poscar.from_file("../searching/log/formal/250125_quinary_bulk_modulus_fcc_" + save_path_name + "/POSCAR_" + str(len(study.trials)-1) + '_{:.2f}'.format(y[-1]),'r')
            structure = poscar.structure
        except:
            structure = input_poscar(element_list, ratio_list, framework)
        structure = re_construct_structure(structure ,element_list, ratio_list, framework)
        
    data = load_data(structure,material_property)
    y_pred =  calculate_y(data, material_property)
    y.append(y_pred)
    
    if len(study.trials) > 1:
        trial = study.best_trial
        y_old = trial.value
    else:
        y_old = 0
    print("best result:{}".format(y_old))
    
    if len(study.trials) > 1 and ((y_pred > 30 and framework == "bcc") or (y_pred > 40 and framework == "fcc")):
        elements = structure.species
        coords = structure.frac_coords
        sorted_indices = sorted(range(len(elements)), key=lambda i: elements[i])
        sorted_elements = [elements[i] for i in sorted_indices]
        sorted_coords = [coords[i] for i in sorted_indices]
        new_structure = Structure(structure.lattice, sorted_elements, sorted_coords)
        
        new_structure.to("../searching/log/formal/250126_quinary_bulk_modulus_fcc_" + save_path_name + "/POSCAR_" + str(len(study.trials)) + '_{:.2f}'.format(y_pred) + "_notmax",fmt = "poscar")
        torch.save(y, "../searching/log/formal/250126_quinary_bulk_modulus_fcc_" + save_path_name + "/y_" + str(len(study.trials)) + '_{:.2f}'.format(y_pred) + "_notmax" +  ".pt")
    
    if len(study.trials) > 1 and (y[-1] > y_old or y[-1] == y_old):
        elements = structure.species
        coords = structure.frac_coords
        sorted_indices = sorted(range(len(elements)), key=lambda i: elements[i])
        sorted_elements = [elements[i] for i in sorted_indices]
        sorted_coords = [coords[i] for i in sorted_indices]
        new_structure = Structure(structure.lattice, sorted_elements, sorted_coords)
        
        new_structure.to("../searching/log/formal/250126_quinary_bulk_modulus_fcc_" + save_path_name + "/POSCAR_" + str(len(study.trials)) + '_{:.2f}'.format(y[-1]),fmt = "poscar")
        torch.save(y, "../searching/log/formal/250126_quinary_bulk_modulus_fcc_" + save_path_name + "/y_" + str(len(study.trials)) + '_{:.2f}'.format(y[-1]) + ".pt")
    return y_pred

if __name__ == '__main__' :
    y = []
    element_num = 4 #quaternary high entropy alloy
    save_path_name = "Fe_total"
    os.mkdir("../searching/log/formal/250126_quinary_bulk_modulus_fcc_" + save_path_name)
    
    # searching elements(quaternary)
    selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Ru","Co","Rh","Pd","Ni","Pt","Cu","Ag","Au","Zn","Ga","Sn","Sb","Bi","La"]#Fe_total
    #selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Sn","Sb","Bi"]#Fe_single
    #selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Ru","Co","Rh","Pd","Ni","Pt","Cu","Ag","Au","Zn","Ga","Sn","Sb","Bi"]#Fe_noble
    #selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Sn","Sb","Bi","Sc","Tc","Ir","Mo","W"]#Fe_mix
    
    #selected_element_list = ["Ca","Sr","Sc","Y","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","Ru","Os","Ir","Pt","Nd","Pm","Eu","Gd","Tb","Dy","Er","Tm","Yb","Lu"]#Zr_total
    #selected_element_list = ["Ca","Sr","Ti","Zr","Hf","Nb","Ta","Mo","W","Re"]#Zr_single
    #selected_element_list = ["Ca","Sr","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","Ru","Os","Ir","Pt"]#Zr_noble
    #selected_element_list = ["Ca","Sr","Sc","Y","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","Nd","Pm","Eu","Gd","Tb","Dy","Er","Tm","Yb","Lu"]#Zr_rare
    #selected_element_list = ["Ca","Sr","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","V","Cr","La","Pr"]#Zr_mix
    
    #selected_element_list = ["Mg","Cr","Mn","Fe","Co","Ni","Cu","Ag","Au","Zn","Ga","Ge","Sn","Sb","Bi"]#Cu_total
    #selected_element_list = ["Mg","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","Sn","Sb","Bi"]#Cu_single
    #selected_element_list = ["Mg","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","Sn","Sb","Bi","Li","Pd","Al","La"]#Cu_mix
    
    #selected_element_list = ["Mg","Cr","Fe","Co","Ni","Cu","Ag","Zn","Ga","Ge","Sn","Sb","Bi"]#Mg_total
    #selected_element_list = ["Mg","Cr","Fe","Co","Ni","Cu","Zn","Ga","Ge","Sn","Sb","Bi"]#Mg_single
    #selected_element_list = ["Mg","Cr","Fe","Co","Ni","Cu","Ag","Zn","Ga","Ge","Sn","Sb","Bi","Li","V","Mn","La","Al"]#Mg_mix
    
    #selected_element_list = ["Li","Ca","Sc","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Ru","Os","Co","Rh","Pd","Ir","Pt","Ag","Au","La"]#Ti_total
    #selected_element_list = ["Li","Ca","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Co"]#Ti_single
    #selected_element_list = ["Li","Ca","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Ru","Os","Co","Rh","Pd","Ir","Pt","Ag","Au"]#Ti_noble
    #selected_element_list = ["Li","Ca","Sc","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Co","La"]#Ti_rare
    #selected_element_list = ["Li","Ca","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Co","Mg","Zr","Hf","Ni","Cu","Sn","Bi","Nd","Pm"]#Ti_mix
    
    all_combinations = list(itertools.combinations(selected_element_list, element_num))
    all_combinations = [" ".join(list(ele)) for ele in all_combinations]
    random.shuffle(all_combinations)
    
    study = optuna.create_study(direction='maximize')#searching mode
    study.optimize(objective, n_trials=2000)
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))