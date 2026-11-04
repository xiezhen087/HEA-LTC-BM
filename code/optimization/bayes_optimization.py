# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:03:14 2024

@author: PC
"""

import optuna
import torch
from pymatgen.io.vasp import Poscar
from torch_geometric.loader import DataLoader
import sys
import pathlib
import random
from pymatgen.core.periodic_table import Element
import itertools
from pymatgen.core import Structure, Lattice
import numpy as np
import os

# --- local modules: models/ and data_processing/ sit next to optimization/ ---
_CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _CODE_DIR.parent
for _p in (_CODE_DIR / "models", _CODE_DIR / "data_processing"):
    sys.path.insert(0, str(_p))
from mixmodel import MixModel
from data_combine import output_data

# ===================== run configuration =====================
# Override through the command-line wrapper code/optimize.py (recommended),
# or edit here directly.
PROPERTY = "thermal_conduct"     # "thermal_conduct" or "bulk_modulus"
ELEMENT_NUM = 4                  # number of components in the alloy
SELECTED_ELEMENTS = ["Li","Mg","Ti","V","Cr","Mn","Fe","Ru","Co","Rh","Pd",
                     "Ni","Pt","Cu","Ag","Au","Zn","Ga","Sn","Sb","Bi","La"]
FRAMEWORKS = ["fcc"]             # any subset of ["fcc", "bcc", "hcp"]
RATIOS = None                    # e.g. [0.25, 0.25, 0.25]; None -> equal split
N_TRIALS = 2000                  # number of TPE trials
OUT_DIR = "./tpe_run"            # output directory for POSCAR logs
TEMPLATE_ROOT = str(_REPO_ROOT / "initial_structure")
MODEL_DIR_BM = str(_REPO_ROOT / "pre_trained_models" / "bulk_modulus" / "result"
                   / "0625_loop_save_time_1_r_square_0.8799")
MODEL_DIR_LTC = str(_REPO_ROOT / "pre_trained_models" / "thermal_conduct" / "result"
                    / "0802_loop_save_time_0_r_square_0.872")
MODEL_FILE_BM = "model_mix_best_bulk_modulus_0625_except_pure_metal_loop_save_time_1_simple.pth"
MODEL_FILE_LTC = "model_mix_best_thermal_conduct_0802_except_pure_metal_loop_save_time_0.pth"
# =============================================================

# pre-trained models are loaded once and cached
_MODEL_CACHE = {}


def _get_model(prop):
    """Lazy-load and cache the Mix-GNN model for 'bulk_modulus' or 'thermal_conduct'."""
    if prop in _MODEL_CACHE:
        return _MODEL_CACHE[prop]
    model_dir, model_file = (MODEL_DIR_BM, MODEL_FILE_BM) if prop == "bulk_modulus" \
        else (MODEL_DIR_LTC, MODEL_FILE_LTC)
    model = MixModel()
    model.load_state_dict(torch.load(os.path.join(model_dir, model_file),
                                     map_location=torch.device('cpu')))
    normalization = torch.load(os.path.join(model_dir, "normalization.pt"))
    model.eval()
    _MODEL_CACHE[prop] = (model, normalization)
    return _MODEL_CACHE[prop]

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
    # pure-metal template per lattice type; species are fully replaced and the
    # lattice constants are recomputed, so the template metal is immaterial
    template = {"fcc": "fcc/POSCAR_Cu_fcc",
                "bcc": "bcc/POSCAR_W_bcc",
                "hcp": "hcp/POSCAR_Zr_hcp"}[framework]
    poscar = Poscar.from_file(os.path.join(TEMPLATE_ROOT, template), 'r')
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
    model, normalization = _get_model(material_property)
    val_loader = DataLoader([data], batch_size= 1,shuffle= False)
    for batch in val_loader:
        with torch.no_grad():
            energy_per_atom = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
    print("Bulk Modulus:{}".format(energy_per_atom[0][0]))
    y = float(energy_per_atom[0][0])*float(normalization[1]) + float(normalization[0])
    print("y:{}".format(y))
    return y

def objective(trial):
    element_num = ELEMENT_NUM  # number of components in the alloy
    material_property = PROPERTY

    elements = trial.suggest_categorical('elements', all_combinations)
    element_list = elements.split(" ")

    # atomic ratios: equal split by default (the last element takes the
    # remainder); set RATIOS in the configuration to fix the first n-1 ratios
    if RATIOS is not None:
        ratio_list = list(RATIOS)
    else:
        ratio_list = [1.0 / element_num] * (element_num - 1)

    # searching framework
    framework = trial.suggest_categorical('framwork', FRAMEWORKS)

    if len(study.trials) == 1:
        structure = input_poscar(element_list, ratio_list, framework)
    else:
        # warm start: mutate the previous trial's structure when available
        try:
            poscar = Poscar.from_file(os.path.join(out_dir, "POSCAR_" + str(len(study.trials)-1) + '_{:.2f}'.format(y[-1])),'r')
            structure = poscar.structure
        except Exception:
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
        
        new_structure.to(os.path.join(out_dir, "POSCAR_" + str(len(study.trials)) + '_{:.2f}'.format(y_pred) + "_notmax"),fmt = "poscar")
        torch.save(y, os.path.join(out_dir, "y_" + str(len(study.trials)) + '_{:.2f}'.format(y_pred) + "_notmax" +  ".pt"))
    
    if len(study.trials) > 1 and (y[-1] > y_old or y[-1] == y_old):
        elements = structure.species
        coords = structure.frac_coords
        sorted_indices = sorted(range(len(elements)), key=lambda i: elements[i])
        sorted_elements = [elements[i] for i in sorted_indices]
        sorted_coords = [coords[i] for i in sorted_indices]
        new_structure = Structure(structure.lattice, sorted_elements, sorted_coords)
        
        new_structure.to(os.path.join(out_dir, "POSCAR_" + str(len(study.trials)) + '_{:.2f}'.format(y[-1])),fmt = "poscar")
        torch.save(y, os.path.join(out_dir, "y_" + str(len(study.trials)) + '_{:.2f}'.format(y[-1]) + ".pt"))
    return y_pred

def main():
    """Single-objective TPE (Bayesian) composition optimization.

    Reads the module-level run configuration (PROPERTY, ELEMENT_NUM,
    SELECTED_ELEMENTS, FRAMEWORKS, RATIOS, N_TRIALS, OUT_DIR; set via
    code/optimize.py or edited above) and logs every evaluated candidate's
    POSCAR and prediction into OUT_DIR.
    """
    global y, study, out_dir, all_combinations
    y = []
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # candidate element pool (the manuscript's element subgroups are kept
    # commented below as presets)
    selected_element_list = SELECTED_ELEMENTS  # default: Fe_total preset
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
    
    all_combinations = list(itertools.combinations(selected_element_list, ELEMENT_NUM))
    all_combinations = [" ".join(list(ele)) for ele in all_combinations]
    random.shuffle(all_combinations)

    study = optuna.create_study(direction='maximize')#searching mode
    study.optimize(objective, n_trials=N_TRIALS)
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    print("Candidate structures logged to:", os.path.abspath(out_dir))

if __name__ == '__main__':
    main()