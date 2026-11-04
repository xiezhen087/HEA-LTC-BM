# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:32:35 2023

@author: Realunknown
"""
import os
import csv
from pymatgen.io.vasp.inputs import Poscar
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import torch
import statistics
from pymatgen.core.periodic_table import Element
from scipy.stats import zscore
import sys
sys.path.append("./")
from features_normalization_for_running import normalization

gas_elements = ["H" ,"He", "N", "O", "F", "Ne", "Cl", "Ar", "Kr","Xe","Rn"]
metal_elements = ["Li","Na","K","Rb","Cs","Fr","Be","Mg","Ca","Sr","Ba","Ra","Sc","Y","Ti","Zr","Hf","Rf","V","Nb","Ta","Db","Cr","Mo","W","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh","Ir","Mt","Ni","Pb","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Ge","Sn","Sb","Bi","Mc","Po","Lv","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"]
magnetic_elements = ["Fe","Co","Ni","Mn","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Sc","Y"]
num_valence_electrons =  {'Ac': 3, 'Ag': 11, 'Al': 3, 'Am': 9, 'Ar': 8, 'As': 5, 'At': 7, 'Au': 11, 'B': 3, 'Ba': 2, 'Be': 2, 'Bi': 5, 'Br': 7, 'C': 4, 'Ca': 2, 'Cd': 12, 'Ce': 4, 'Cl': 7, 'Cm': 10, 'Co': 9, 'Cr': 6, 'Cs': 1, 'Cu': 11, 'Dy': 12, 'Er': 14, 'Eu': 9, 'F': 7, 'Fe': 8, 'Fr': 1, 'Ga': 3, 'Gd': 10, 'Ge': 4, 'H': 1, 'He': 2, 'Hf': 4, 'Hg': 12, 'Ho': 13, 'I': 7, 'In': 3, 'Ir': 9, 'K': 1, 'Kr': 8, 'La': 3, 'Li': 1, 'Lu': 3, 'Mg': 2, 'Mn': 7, 'Mo': 6, 'N': 5, 'Na': 1, 'Nb': 5, 'Nd': 6, 'Ne': 8, 'Ni': 10, 'Np': 7, 'O': 6, 'Os': 8, 'P': 5, 'Pa': 5, 'Pb': 4, 'Pd': 10, 'Pm': 7, 'Po': 6, 'Pr': 5, 'Pt': 10, 'Pu': 8, 'Ra': 2, 'Rb': 1, 'Re': 7, 'Rh': 9, 'Rn': 8, 'Ru': 8, 'S': 6, 'Sb': 5, 'Sc': 3, 'Se': 6, 'Si': 4, 'Sm': 8, 'Sn': 4, 'Sr': 2, 'Ta': 5, 'Tb': 11, 'Tc': 7, 'Te': 6, 'Th': 4, 'Ti': 4, 'Tl': 3, 'Tm': 15, 'U': 6, 'V': 5, 'W': 6, 'Xe': 8, 'Y': 3, 'Yb': 16, 'Zn': 12, 'Zr': 4}
#space_vs_point_group = {"C1":"p1","C2":"p2"}

#list crystals
def get_crystal_list(order_path):
    order = []
    with open(order_path, encoding='utf-8') as f:
        for row in csv.reader(f, skipinitialspace=True):
            order.append(row[0])
    crystal_mp_id_list = os.listdir(label_path)
    crystal_mp_id_list = [mpid for mpid in crystal_mp_id_list]
    crystal_mp_id_list_ordered = []
    for Num in order:
        crystal_mp_id_list_ordered.append(crystal_mp_id_list[(crystal_mp_id_list.index(Num))])#bm
        #crystal_mp_id_list_ordered.append(crystal_mp_id_list[(crystal_mp_id_list.index("auid-" + Num))])#tc
    return crystal_mp_id_list_ordered

#Operation
def get_num_kpt(mp_id):
    num_kpt = []
    kpt_lines = []
    with open(data_path + mp_id + '-20180417' + '/' +  'KPOINTS-force', 'r') as f:
        for line in f:
            kpt_lines.append(line)
    kpts = kpt_lines[3].split(" ")
    kpts = list(filter(None, kpts))
    kpts = [int(k) for k in kpts]
    my_kpts = np.array(kpts)
    num_kpt = np.prod(my_kpts)
    return num_kpt

#Crystal structure
def get_lattice_type(structure):
    '''
    This function is used to resolve the lattice type of the crystal.

    Returns
    -------
    structure : 
        the class of Pymatgen's crystal structures.
    The meaning of classes of different lattice types:
        0: cubic.
        1: orthorhombic.
        2: monoclinic.
        3: tetragonal.
        4: rhombohedral.
        5: hexagonal.
        6: others.

    '''
    analyzer = SpacegroupAnalyzer(structure)
    crystal_type = analyzer.get_lattice_type()
    if crystal_type == "cubic":
        lattice_type = 0
    elif crystal_type == "orthorhombic":
        lattice_type = 1
    elif crystal_type == "monoclinic":
        lattice_type = 2
    elif crystal_type == "tetragonal":
        lattice_type = 3
    elif crystal_type == "rhombohedral":
        lattice_type = 4
    elif crystal_type == "hexagonal":
        lattice_type = 5
    else:
        lattice_type = 6
    return lattice_type

def get_volume_pre_atom(structure):
    sites = structure.sites
    total_atoms = len(sites)
    cell_volume = structure.volume
    volume_pre_atom = cell_volume/total_atoms
    return volume_pre_atom

def get_volume_pre_cell(structure):
    volume_pre_cell = structure.volume
    return volume_pre_cell

def get_variance_of_atom_radius(structure):
    sites = structure.sites
    atomic_radius = []
    for site in sites:
        element = str(site.specie)
        if Element(element).atomic_radius != None:
            atomic_radius.append(float(Element(element).atomic_radius))
        else:
            atomic_radius.append(1.0)
    if len(atomic_radius) > 1:
        variance_of_atom_radius = statistics.variance(atomic_radius)
    else:
        variance_of_atom_radius = 0.0
    return variance_of_atom_radius

def get_extreme_atom_radius_mean_z_score(structure):
    sites = structure.sites
    atomic_radius = []
    for site in sites:
        element = str(site.specie)
        if Element(element).atomic_radius != None:
            atomic_radius.append(float(Element(element).atomic_radius))
        else:
            atomic_radius.append(1.0)
    if len(set(atomic_radius)) == 1:
        extreme_atom_radius_z_score = [0]*len(atomic_radius)
    else:
        extreme_atom_radius_z_score = zscore(atomic_radius)
    ex_z_score = []
    for extreme_z_score in extreme_atom_radius_z_score:
        if extreme_z_score > 0 and len(ex_z_score) < 10:
            ex_z_score.append(extreme_z_score)
        else:
            ex_z_score = [0]
    extreme_atom_radius_mean_z_score = sum(ex_z_score)/len(ex_z_score)
    return extreme_atom_radius_mean_z_score

#Elemental_composition
def get_number_of_species(structure):
    elements = set([site.specie for site in structure])
    number_of_species = len(elements)
    return number_of_species

def get_number_of_atom_per_cell(structure):
    sites = structure.sites
    atom_per_cell= len(sites)
    return atom_per_cell

def is_gas(structure):
    is_gas = 1
    elements = set([str(site.specie) for site in structure])
    for ele in elements:
        if ele not in gas_elements:
            is_gas = 0
    return is_gas

def is_metal(structure):
    elements = [str(site.specie) for site in structure]
    not_metal = []
    for ele in elements:
        if str(ele) not in metal_elements:
            not_metal.append(1)
    if len(not_metal)/len(elements) > 0.1:
        is_metal = 0
    else:
        is_metal = 1
    return is_metal

def get_metal_elements_proportion(structure):
    elements = [str(site.specie) for site in structure]
    in_metal = []
    for ele in elements:
        if str(ele) in metal_elements:
            in_metal.append(1)
    metal_elements_proportion = len(in_metal)/len(elements)
    return metal_elements_proportion

def get_magnetic_elements_proportion(structure):
    elements = [str(site.specie) for site in structure]
    in_magnetic = []
    for ele in elements:
        if str(ele) in magnetic_elements:
            in_magnetic.append(1)
    magnetic_elements_proportion = len(in_magnetic)/len(elements)
    return magnetic_elements_proportion

#Characters_weighted
def get_cw_atomic_number(structure):
    atomic_number = []
    for site in structure:
        element = site.specie
        atomic_number.append(element.Z)
    cw_atomic_number = sum([ele/len(atomic_number) for ele in atomic_number])
    return cw_atomic_number

def get_cw_valence_electrons(structure):
    valence_electrons = []
    for site in structure:
        element = site.specie
        try:
            valence_electrons.append(sum(list(Element(element.symbol).valence)))
        except:
            valence_electrons.append(0)
    cw_valence_electrons = sum([ele/len(valence_electrons) for ele in valence_electrons])
    if type(cw_valence_electrons) is not float:
        cw_valence_electrons = 0.0
    return cw_valence_electrons

def get_cw_atomic_mass(structure):
    atomic_mass = []
    for site in structure:
        element = site.specie
        atomic_mass.append(Element(element.symbol).atomic_mass)
    cw_atomic_mass = sum([ele/len(atomic_mass) for ele in atomic_mass])
    return cw_atomic_mass

def get_cw_group(structure):
    group = []
    for site in structure:
        element = site.specie
        group.append(Element(element.symbol).group)
    cw_group = sum([ele/len(group) for ele in group])
    return cw_group

def get_cw_period(structure):
    period = []
    for site in structure:
        element = site.specie
        period.append(Element(element.symbol).row)
    cw_period = sum([ele/len(period) for ele in period])
    return cw_period

def get_cw_electronegativity(structure):
    electronegativity = []
    for site in structure:
        element = site.specie
        electronegativity.append(Element(element.symbol).X)
    cw_electronegativity = sum([ele/len(electronegativity) for ele in electronegativity])
    if type(cw_electronegativity) is not float:
        cw_electronegativity = 0.0
    return cw_electronegativity

def get_cw_boiling_point(structure):
    boiling_point = []
    for site in structure:
        element = site.specie
        if Element(element.symbol).boiling_point != None:
            boiling_point.append(float(Element(element.symbol).boiling_point))
        else:
            boiling_point.append(0)
    cw_boiling_point = sum([ele/len(boiling_point) for ele in boiling_point])
    return cw_boiling_point

def get_cw_melting_point(structure):
    melting_point = []
    for site in structure:
        element = site.specie
        melting_point.append(float(Element(element.symbol).melting_point))
    cw_melting_point = sum([ele/len(melting_point) for ele in melting_point])
    return cw_melting_point

def get_cw_van_der_Waals_radii(structure):
    van_der_Waals_radii = []
    for site in structure:
        element = site.specie
        van_der_Waals_radii.append(Element(element.symbol).van_der_waals_radius)
    cw_van_der_Waals_radii = sum([ele/len(van_der_Waals_radii) for ele in van_der_Waals_radii])
    return cw_van_der_Waals_radii

def get_cw_atomic_radius_calculated(structure):
    atomic_radius_calculated = []
    for site in structure:
        element = site.specie
        if Element(element.symbol).atomic_radius_calculated != None:
            atomic_radius_calculated.append(Element(element.symbol).atomic_radius_calculated)
        else:
            atomic_radius_calculated.append(1.0)
    cw_atomic_radius_calculated = sum([ele/len(atomic_radius_calculated ) for ele in atomic_radius_calculated ])
    return cw_atomic_radius_calculated

def get_cw_electrical_resistivity(structure):
    electrical_resistivity = []
    for site in structure:
        element = site.specie
        if Element(element.symbol).electrical_resistivity != None:
            electrical_resistivity.append(Element(element.symbol).electrical_resistivity)
        else:
            electrical_resistivity.append(0)
    cw_electrical_resistivity = sum([ele/len(electrical_resistivity) for ele in electrical_resistivity ])
    return cw_electrical_resistivity

def get_cw_velocity_of_sound(structure):
    velocity_of_sound = []
    for site in structure:
        element = site.specie
        if Element(element.symbol).velocity_of_sound != None:
             velocity_of_sound.append(Element(element.symbol).velocity_of_sound)
        else:
            velocity_of_sound.append(1.0)
    cw_velocity_of_sound = sum([ele/len(velocity_of_sound) for ele in velocity_of_sound])
    return cw_velocity_of_sound

def get_cw_thermal_conductivity(structure):
    thermal_conductivity = []
    for site in structure:
        element = site.specie
        thermal_conductivity.append(Element(element.symbol).thermal_conductivity)
    cw_thermal_conductivity = sum([ele/len(thermal_conductivity) for ele in thermal_conductivity])
    return cw_thermal_conductivity

def get_cw_bulk_modulus(structure):
    bulk_modulus = []
    for site in structure:
        element = site.specie
        if Element(element.symbol).bulk_modulus != None:
            bulk_modulus.append(Element(element.symbol).bulk_modulus)
        else:
            bulk_modulus.append(0)
    cw_bulk_modulus = sum([ele/len(bulk_modulus) for ele in bulk_modulus])
    return cw_bulk_modulus

def get_cw_youngs_modulus(structure):
    youngs_modulus = []
    for site in structure:
        element = site.specie
        if Element(element.symbol).youngs_modulus != None:
            youngs_modulus.append(Element(element.symbol).youngs_modulus)
        else:
            youngs_modulus.append(0)
    cw_youngs_modulus = sum([ele/len(youngs_modulus) for ele in youngs_modulus])
    return cw_youngs_modulus

def get_cw_coefficient_of_linear_thermal_expansion(structure):
    coefficient_of_linear_thermal_expansion = []
    for site in structure:
        element = site.specie
        if Element(element.symbol).coefficient_of_linear_thermal_expansion != None:
            coefficient_of_linear_thermal_expansion.append(Element(element.symbol).coefficient_of_linear_thermal_expansion)
        else:
            coefficient_of_linear_thermal_expansion.append(0)
    cw_coefficient_of_linear_thermal_expansion = sum([ele/len(coefficient_of_linear_thermal_expansion) for ele in coefficient_of_linear_thermal_expansion])
    return cw_coefficient_of_linear_thermal_expansion

def get_cw_ionization_energies(structure):
    ionization_energies = []
    for site in structure:
        element = site.specie
        ionization_energies.append(Element(element.symbol).ionization_energies[0])
    ionization_energies = sum([ele/len(ionization_energies) for ele in ionization_energies])
    return ionization_energies

#for thermal conduct
def get_density(structure):
    atoms_mass = []
    for site in structure:
        element = site.specie
        atoms_mass.append(Element(element.symbol).atomic_mass)
    volume = structure.volume
    density = sum(atoms_mass)/volume
    return density

def get_space_group(structure):
    analyzer = SpacegroupAnalyzer(structure)
    space_group = analyzer.get_space_group_number()
    return space_group

def get_mlp_data(structure,material_property):
    x_line = []
    x_line.append("run_data")
    density = get_density(structure)#Thermal_conduct
    x_line.append(density)
    space_group = get_space_group(structure)
    x_line.append(space_group)
    
    lattice_type = get_lattice_type(structure)
    x_line.append(lattice_type)
    volume_pre_atom = get_volume_pre_atom(structure)
    x_line.append(volume_pre_atom)
    volume_pre_cell = get_volume_pre_cell(structure)
    x_line.append(volume_pre_cell)
    variance_of_atom_radius = get_variance_of_atom_radius(structure)
    x_line.append(variance_of_atom_radius)
    extreme_atom_radius_mean_z_score = get_extreme_atom_radius_mean_z_score(structure)
    x_line.append(extreme_atom_radius_mean_z_score)

    number_of_species = get_number_of_species(structure)
    x_line.append(number_of_species)
    number_of_atom_per_cell = get_number_of_atom_per_cell(structure)
    x_line.append(number_of_atom_per_cell)
    Is_gas = is_gas(structure)
    x_line.append(Is_gas)
    Is_metal = is_metal(structure)
    x_line.append(Is_metal)
    metal_elements_proportion = get_metal_elements_proportion(structure)
    x_line.append(metal_elements_proportion)
    magnetic_elements_proportion = get_magnetic_elements_proportion(structure)
    x_line.append(magnetic_elements_proportion)
    
    cw_atomic_number = get_cw_atomic_number(structure)
    x_line.append(cw_atomic_number)
    cw_valence_electrons = get_cw_valence_electrons(structure)
    x_line.append(cw_valence_electrons)
    cw_atomic_mass = get_cw_atomic_mass(structure)
    x_line.append(cw_atomic_mass)
    cw_group = get_cw_group(structure)
    x_line.append(cw_group)
    cw_period = get_cw_period(structure)
    x_line.append(cw_period)
    cw_electronegativity = get_cw_electronegativity(structure)
    x_line.append(cw_electronegativity)
    cw_boiling_point = get_cw_boiling_point(structure)
    x_line.append(cw_boiling_point)
    cw_melting_point = get_cw_melting_point(structure)
    x_line.append(cw_melting_point)
    cw_van_der_Waals_radii = get_cw_van_der_Waals_radii(structure)
    x_line.append(cw_van_der_Waals_radii)
    cw_atomic_radius_calculated = get_cw_atomic_radius_calculated(structure)
    x_line.append(cw_atomic_radius_calculated)
    
    cw_electrical_resistivity = get_cw_electrical_resistivity(structure)
    x_line.append(cw_electrical_resistivity)
    cw_velocity_of_sound = get_cw_velocity_of_sound(structure)
    x_line.append(cw_velocity_of_sound)
    cw_thermal_conductivity = get_cw_thermal_conductivity(structure)
    x_line.append(cw_thermal_conductivity)
    cw_bulk_modulus = get_cw_bulk_modulus(structure)
    x_line.append(cw_bulk_modulus)
    cw_youngs_modulus = get_cw_youngs_modulus(structure)
    x_line.append(cw_youngs_modulus)
    cw_coefficient_of_linear_thermal_expansion = get_cw_coefficient_of_linear_thermal_expansion(structure)
    x_line.append(cw_coefficient_of_linear_thermal_expansion)
    cw_ionization_energies= get_cw_ionization_energies(structure)
    x_line.append(cw_ionization_energies)
    target = 0.0
    x_line.append(target)
    x_line = normalization(x_line,material_property)
    return x_line

if __name__ == '__main__' :
    material_property = "bulk_modulus"
    #order_path = './elasiticity/bulk_modulus_index.csv'#Bulk Modulus
    order_path = './thermal_conduct/thermal_conductivity_index.csv'#Thermal Conductivity
    #data_path = "./data/data_voronoi/"
    data_path = "./data/thermo_conduct_struc_5k/"#Thermal Conductivity  # TODO: set your local data path
    #label_path = "./data/elasticity_1w/"#Bulk Modulus
    label_path = "./data/thermo_conduct_5k/"#Thermal Conductivity  # TODO: set your local data path
    crystal_list = get_crystal_list(order_path)
    poscar = Poscar.from_file(data_path + "mp-1000" + '/CONTCAR')#Thermal Conductivity
    structure = poscar.structure
    x_line = get_mlp_data(structure,material_property)