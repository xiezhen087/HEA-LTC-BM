# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 20:59:09 2025

@author: Realunknown
"""
import os
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
import numpy as np
import random
import shutil
import torch
import sys
import pathlib
_CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _CODE_DIR.parent
for _p in (_CODE_DIR / "models", _CODE_DIR / "data_processing"):
    sys.path.insert(0, str(_p))
from torch_geometric.loader import DataLoader

from mixmodel import MixModel
from data_combine import output_data
import sys
sys.path.append("./")

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
    return vec

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

def modify_lattice_constant(structure,framework):
    lattice_constant = [0]*3
    species = structure.species
    num_species = len(set(species))
    species_type = list(set(species))
    species_radius_type = [ele.atomic_radius_calculated for ele in species_type]
    try:
        sum(species_radius_type) > 0
    except:
        species_radius_type = [3.5,3.5]
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

def generate_new_structure(structure ,element_list, ratio_list, framework):    
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

def load_data(structure, material_property):
    data = output_data(structure,material_property)
    return data

def calculate_y(data, material_property):
    if  material_property == "bulk_modulus":
        model_path = os.path.join(_REPO_ROOT, "pre_trained_models", "bulk_modulus", "result", "0625_loop_save_time_1_r_square_0.8799") + os.sep  # shipped model
        model = MixModel()
        model.load_state_dict(torch.load(model_path + "model_mix_best_" + material_property + "_0625_except_pure_metal_loop_save_time_1_simple.pth" ,map_location=torch.device('cpu')))
    elif  material_property == "thermal_conduct":
        model_path = os.path.join(_REPO_ROOT, "pre_trained_models", "thermal_conduct", "result", "0802_loop_save_time_0_r_square_0.872") + os.sep  # shipped model
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

if __name__ == '__main__' :
    material_property = "bulk_modulus"
    framework = "bcc"
    
    file_path = "./data/less_than_0.045/"  # TODO: set your local data path
    file_name = "POSCAR_1731_54.85_notmax"
    poscar = Poscar.from_file(file_path + file_name,'r')
    structure = poscar.structure
    initial_vec = calculate_vec(structure)
    print("Initial VEC:{}".format(initial_vec))
    initial_delta = calculate_delta(structure)
    print("Initial delta:{}".format(initial_delta))
    data = load_data(structure, material_property)
    initial_predicted_value = calculate_y(data, material_property)
    print("Initial Predicted_Value:{}".format(initial_predicted_value))
    
    #Generate New Structure
    elements = [atom.specie.symbol for atom in structure]
    element_list = sorted({x:0 for x in elements}, key=lambda x: (elements.count(x), elements.index(x)))
    print("Initial elements:{}".format(element_list))
    vec_element_list = [num_valence_electrons[ele] for ele in element_list]
    print("Initial VEC of elements:{}".format(vec_element_list))
    initial_ratio_list = [elements.count(elements[0])/len(elements),elements.count(elements[1])/len(elements),elements.count(elements[2])/len(elements),elements.count(elements[3])/len(elements)]
    new_ratio_list = [0.12,0.2,0.2,0.2]
    structure_new = generate_new_structure(structure ,element_list, new_ratio_list, framework)
    new_vec = calculate_vec(structure_new)
    print("New VEC:{}".format(new_vec))
    new_delta = calculate_delta(structure)
    print("New delta:{}".format(new_delta))
    new_data = load_data(structure_new, material_property)
    new_predicted_value = calculate_y(new_data, material_property)
    print("New Predicted_Value:{}".format(new_predicted_value))
    
    #clear poscar
    elements_poscar = structure_new.species
    coords = structure_new.frac_coords
    sorted_indices = sorted(range(len(elements_poscar)), key=lambda i: elements_poscar[i])
    sorted_elements = [elements_poscar[i] for i in sorted_indices]
    sorted_coords = [coords[i] for i in sorted_indices]
    structure_output = Structure(structure_new.lattice, sorted_elements, sorted_coords)
    
    elements_new = [atom.specie.symbol for atom in structure_new]
    num_atoms = [elements_new.count(element_list[i]) for i in range(len(element_list))]
    output_file_path = "./data/efficient_data_non_equiatoms/"  # TODO: set your local data path
    system = element_list[0] + str(num_atoms[0]) + element_list[1] + str(num_atoms[1]) + element_list[2] + str(num_atoms[2]) + element_list[3] + str(num_atoms[3]) + element_list[4] + str(num_atoms[4])
    #structure_output.to(output_file_path + "/POSCAR_" + system + '_{:.2f}'.format(new_predicted_value),fmt = "poscar")