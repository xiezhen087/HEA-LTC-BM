# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 07:54:53 2022

@author: Realunknown
"""
from pymatgen.io.vasp import Poscar
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.analysis.local_env import VoronoiNN,CrystalNN
import torch
from torch_geometric.data import Data
from scipy.constants import Avogadro
import math
import numpy as np
import csv
import os

num_valence_electrons =  {'Ac': 3, 'Ag': 11, 'Al': 3, 'Am': 9, 'Ar': 8, 'As': 5, 'At': 7, 'Au': 11, 'B': 3, 'Ba': 2, 'Be': 2, 'Bi': 5, 'Br': 7, 'C': 4, 'Ca': 2, 'Cd': 12, 'Ce': 4, 'Cl': 7, 'Cm': 10, 'Co': 9, 'Cr': 6, 'Cs': 1, 'Cu': 11, 'Dy': 12, 'Er': 14, 'Eu': 9, 'F': 7, 'Fe': 8, 'Fr': 1, 'Ga': 3, 'Gd': 10, 'Ge': 4, 'H': 1, 'He': 2, 'Hf': 4, 'Hg': 12, 'Ho': 13, 'I': 7, 'In': 3, 'Ir': 9, 'K': 1, 'Kr': 8, 'La': 3, 'Li': 1, 'Lu': 3, 'Mg': 2, 'Mn': 7, 'Mo': 6, 'N': 5, 'Na': 1, 'Nb': 5, 'Nd': 6, 'Ne': 8, 'Ni': 10, 'Np': 7, 'O': 6, 'Os': 8, 'P': 5, 'Pa': 5, 'Pb': 4, 'Pd': 10, 'Pm': 7, 'Po': 6, 'Pr': 5, 'Pt': 10, 'Pu': 8, 'Ra': 2, 'Rb': 1, 'Re': 7, 'Rh': 9, 'Rn': 8, 'Ru': 8, 'S': 6, 'Sb': 5, 'Sc': 3, 'Se': 6, 'Si': 4, 'Sm': 8, 'Sn': 4, 'Sr': 2, 'Ta': 5, 'Tb': 11, 'Tc': 7, 'Te': 6, 'Th': 4, 'Ti': 4, 'Tl': 3, 'Tm': 15, 'U': 6, 'V': 5, 'W': 6, 'Xe': 8, 'Y': 3, 'Yb': 16, 'Zn': 12, 'Zr': 4}

#list crystals
def get_crystal_list(order_path):
    order = []
    with open(order_path, encoding='utf-8') as f:
        for row in csv.reader(f, skipinitialspace=True):
            order.append(row[0])
    crystal_mp_id_list = os.listdir(data_path)
    crystal_mp_id_list = [mpid for mpid in crystal_mp_id_list]
    crystal_mp_id_list_ordered = []
    for Num in order:
        crystal_mp_id_list_ordered.append(crystal_mp_id_list[(crystal_mp_id_list.index(Num ))])
    return crystal_mp_id_list_ordered

def get_x_Z(atom):
    r"""
    This function is used to get the atomic number Z.   
    """
    Z = Element(atom.specie.symbol).Z
    return Z

def get_x_mass(atom):
    r"""
    This function is used to get the atomic mass.   
    """
    mass = int(Element(atom.specie.symbol).atomic_mass)
    return torch.tensor([mass])

def get_x_Gnum(atom):
    r"""
    This function is used to get the group number.   
    """
    Group_num = Element(atom.specie.symbol).group
    return Group_num
   
def get_x_period_num(atom):
    r"""
    This function is used to get the period number.  
    """
    period_num = Element(atom.specie.symbol).row
    return period_num

def get_x_electronegativity(atom):
    r"""
    This function is used to get the electronegativity.  
    """
    e_n = Element(atom.specie.symbol).X
    if 0 < e_n <= 0.4: 
       elec_nega = 0
    elif 0.4 < e_n <= 0.8:
        elec_nega = 1
    elif 0.8 < e_n <= 1.2:
       elec_nega = 2
    elif 1.2 < e_n <= 1.6:
        elec_nega = 3
    elif 1.6 < e_n <= 2.0:
        elec_nega = 4
    elif 2.0 < e_n <= 2.4:
        elec_nega = 5
    elif 2.4 < e_n <= 2.8:
        elec_nega = 6
    elif 2.8 < e_n <= 3.2:
        elec_nega = 7
    elif 3.2 < e_n <= 3.6:
        elec_nega = 8
    elif 3.6 < e_n <= 4.0:
        elec_nega = 9
    else:
        elec_nega = 10
    return elec_nega

def get_x_covalent_radius(atom):
    r"""
    This function is used to get the covalent radius.  
    Beatriz C. et al. Dalton Trans. 2008, 2832-2838. DOI: 10.1039/b801115j
    paper
    """
    C_r = CovalentRadius.radius[atom.specie.symbol]
    if 0 < C_r <= 0.25: 
        C_radius = 0
    elif 0.25 < C_r <= 0.50:
        C_radius = 1
    elif 0.50 < C_r <= 0.75:
        C_radius = 2
    elif 0.75 < C_r <= 1.00:
        C_radius = 3
    elif 1.00 < C_r <= 1.25:
        C_radius = 4
    elif 1.25 < C_r <= 1.50:
        C_radius = 5
    elif 1.50 < C_r <= 1.75:
        C_radius = 6
    elif 1.75 < C_r <= 2.00:
        C_radius = 7
    elif 2.00 < C_r <= 2.25:
        C_radius = 8
    elif 2.25 < C_r <= 2.50:
        C_radius = 9
    elif 2.50 < C_r :
        C_radius = 10
    return C_radius

def get_x_valence_electrons(atom):
    r"""
    This function is used to get the number of valence_electrons. 
    """
    V_elec = num_valence_electrons[atom.specie.symbol]
    return V_elec

def get_x_first_ionization_energy(atom):
    r"""
    This function is used to get the first ionization energy. 
    Values are from the NIST Atomic Spectra Database.(eV)
    """
    fie = math.log(Element(atom.specie.symbol).ionization_energies[0])
    if 0 < fie  <= 1.3: 
        first_ion_energy = 0
    elif 1.3 < fie  <= 1.5:
        first_ion_energy = 1
    elif 1.5 < fie  <= 1.7:
        first_ion_energy = 2
    elif 1.7 < fie  <= 1.9:
        first_ion_energy = 3
    elif 1.9 < fie  <= 2.1:
        first_ion_energy = 4
    elif 2.1 < fie  <= 2.3:
        first_ion_energy = 5
    elif 2.3 < fie  <= 2.5:
        first_ion_energy = 6
    elif 2.5 < fie  <= 2.7:
        first_ion_energy = 7
    elif 2.7 < fie  <= 2.9:
        first_ion_energy = 8
    elif 2.9 < fie  <= 3.1:
        first_ion_energy = 9
    elif 3.1 < fie  <= 3.3:
        first_ion_energy = 10
    elif 3.3 < fie  :
        first_ion_energy = 11
    return first_ion_energy

def get_x_electron_affinity(atom):
    r"""
    This function is used to get the electron affinity. 
    """
    ea = Element(atom.specie.symbol).electron_affinity
    if ea  <= -5.0: 
        affinity = 0
    elif -5.0 < ea  <= -4.0:
        affinity = 1
    elif -4.0 < ea  <= -3.0:
        affinity = 2
    elif -3.0 < ea  <= -2.0:
        affinity = 3
    elif -2.0 < ea  <=  -1.0:
        affinity = 4
    elif -1.0 < ea  <= 0.0:
        affinity = 5
    elif 0.0 < ea  <= 1.0:
        affinity = 6
    elif 1.0 < ea  <= 2.0:
        affinity = 7
    elif 2.0 < ea  <= 3.0:
        affinity = 8
    elif 3.0 < ea  <= 4.0:
        affinity = 9
    elif 4.0 < ea  :
        affinity = 10
    return affinity

def get_x_block(atom):
    r"""
    This function is used to get the element block. 
    """
    eb = Element(atom.specie.symbol).block
    if eb == 's': 
        block = 0
    elif eb == 'p':
        block = 1
    elif eb == 'd':
        block = 2
    elif eb == 'f':
        block = 3
    return block

def get_x_atomic_volume(atom):
    r"""
    This function is used to get the atomic volume. 
    """
    try:
        volume = math.log((Element(atom.specie.symbol).atomic_radius*1e-8)** 3 * 4 / 3 * 3.141592653589793*Avogadro)
    except:
        volume = math.log((Element(atom.specie.symbol).atomic_radius_calculated*1e-8)** 3 * 4 / 3 * 3.141592653589793*Avogadro)
    atom_volume = 0
    if 0 < volume <= 0.4: 
       atom_volume = 0
    elif 0.4 < volume <= 0.8:
        atom_volume = 1
    elif 0.8 < volume <= 1.2:
       atom_volume = 2
    elif 1.2 < volume <= 1.6:
        atom_volume = 3
    elif 1.6 < volume <= 2.0:
        atom_volume = 4
    elif 2.0 < volume <= 2.4:
        atom_volume = 5
    elif 2.4 < volume <= 2.8:
        atom_volume = 6
    elif 2.8 < volume <= 3.2:
        atom_volume = 7
    elif 3.2 < volume <= 3.6:
        atom_volume = 8
    elif 3.6 < volume <= 4.0:
        atom_volume = 9
    elif 4.0 < volume :
        atom_volume = 10
    return atom_volume

def load_features(atom_fea):
    num_valence_electrons =  {'Ac': 3, 'Ag': 11, 'Al': 3, 'Am': 9, 'Ar': 8, 'As': 5, 'At': 7, 'Au': 11, 'B': 3, 'Ba': 2, 'Be': 2, 'Bi': 5, 'Br': 7, 'C': 4, 'Ca': 2, 'Cd': 12, 'Ce': 4, 'Cl': 7, 'Cm': 10, 'Co': 9, 'Cr': 6, 'Cs': 1, 'Cu': 11, 'Dy': 12, 'Er': 14, 'Eu': 9, 'F': 7, 'Fe': 8, 'Fr': 1, 'Ga': 3, 'Gd': 10, 'Ge': 4, 'H': 1, 'He': 2, 'Hf': 4, 'Hg': 12, 'Ho': 13, 'I': 7, 'In': 3, 'Ir': 9, 'K': 1, 'Kr': 8, 'La': 3, 'Li': 1, 'Lu': 3, 'Mg': 2, 'Mn': 7, 'Mo': 6, 'N': 5, 'Na': 1, 'Nb': 5, 'Nd': 6, 'Ne': 8, 'Ni': 10, 'Np': 7, 'O': 6, 'Os': 8, 'P': 5, 'Pa': 5, 'Pb': 4, 'Pd': 10, 'Pm': 7, 'Po': 6, 'Pr': 5, 'Pt': 10, 'Pu': 8, 'Ra': 2, 'Rb': 1, 'Re': 7, 'Rh': 9, 'Rn': 8, 'Ru': 8, 'S': 6, 'Sb': 5, 'Sc': 3, 'Se': 6, 'Si': 4, 'Sm': 8, 'Sn': 4, 'Sr': 2, 'Ta': 5, 'Tb': 11, 'Tc': 7, 'Te': 6, 'Th': 4, 'Ti': 4, 'Tl': 3, 'Tm': 15, 'U': 6, 'V': 5, 'W': 6, 'Xe': 8, 'Y': 3, 'Yb': 16, 'Zn': 12, 'Zr': 4}
    atom_index = atom_fea[0]
    element = Element.from_Z(atom_index)
    electronegativity = Element.from_Z(atom_index).X
    if type(electronegativity) is not float:
        electronegativity = 0.0
    atomic_mass_re = 1/(element.atomic_mass)
    num_valence = (num_valence_electrons[element.symbol]-1)/15#min-max normalization
    if type(num_valence) is not float:
        num_valence = 0.0
    atom_fea = atom_fea + [electronegativity,atomic_mass_re,num_valence]#magnify force number
    return atom_fea

def get_atom_features(structure):
    r"""
    This function is used to construct feature matrix.
    
    """
    #ordering_paras = torch.load(os.path.join(file_path + '/', file_name + '-order_characters.pt'))
    x = []
    for i in range(len(structure)):
        atom_fea = []
        atom_fea.append(get_x_Z(structure[i]))
        atom_fea.append(get_x_Gnum(structure[i]))
        atom_fea.append(get_x_period_num(structure[i]))
        atom_fea.append(get_x_electronegativity(structure[i]))
        atom_fea.append(get_x_covalent_radius(structure[i]))
        atom_fea.append(get_x_valence_electrons(structure[i]))
        atom_fea.append(get_x_first_ionization_energy(structure[i]))
        atom_fea.append(get_x_electron_affinity(structure[i]))
        atom_fea.append(get_x_block(structure[i]))
        atom_fea.append(get_x_atomic_volume(structure[i]))
        atom_fea = load_features(atom_fea)
        x.append(atom_fea)
    return torch.Tensor(x)

def get_edge_index(structure):
    r"""
    This function imports the adjacency matrix of the structure.
    """
    edge_index = []
    total_atoms = structure.num_sites
    source_index = []
    target_index = []
    for i in range(total_atoms):
        cn = CrystalNN()
        try:
            nearneighbor = cn.get_nn_info(structure, i)
            for j in range(len(nearneighbor)):
                source_index.append(i)
                target_index.append(nearneighbor[j]['site_index'])
        except:
            pass
    edge_index.append(source_index)
    edge_index.append(target_index)
    return torch.tensor(edge_index)

def get_edge_attr(structure):
    r"""
    This function imports chemical bond long data from files.
    """
    edge_attr = []
    radius = 8
    dmin = 0
    dmax = radius
    step = 0.2
    edge_filter = np.arange(dmin, dmax+step, step)
    total_atoms = structure.num_sites
    edge_length = []
    for i in range(total_atoms):
        cn = CrystalNN()
        try:
            nearneighbor = cn.get_nn_info(structure, i)
            for j in range(len(nearneighbor)):
                edge_length.append(nearneighbor[j]['site'].nn_distance)
        except:
            pass
    #Chemical bond characteristics are divided into ten categories according to different lengths
    edge_length = np.array(edge_length)
    edge_attr = np.exp(-(edge_length[..., np.newaxis] - edge_filter)**2 /
                      step**2)
    return torch.tensor(edge_attr)

def get_structures(file_path):
    r"""
    In this function, the pymatgen module is used to import the target POSCAR structure file and convert it into a structure object.
    
    """
    poscar = Poscar.from_file(file_path + "POSCAR",'r')
    #poscar = Poscar.from_file(file_path + "CONTCAR",'r')#Thermo_conduct.
    structure = poscar.structure
    return structure

def get_cgcnn_data(structure):
    x = get_atom_features(structure)
    edge_index = get_edge_index(structure)
    edge_attr = get_edge_attr(structure)

    y = torch.tensor(0.0)
    data = Data(x = x,edge_index = edge_index,edge_attr = edge_attr,y = y)
    return data

if __name__ == '__main__' :
    r"""
    This script is used to construct the graph data structure of the CGCNN model. 
    Part of the data was imported from an existing database, such as POSCAR files, edge_index of vesta, edge_index of voronoi, labels, edge_attr.
    
    ----------
    Contents in Data:
    x: The features of atom nodes.(Tensor: N * atom_fea_len)
    edge_index: Adjacency matrix represented by COO form (index of source atoms and index of target atoms,Tensor: 2 * edge_num)
    edge_attr: The features of bonds.(Tensor: bond_fea_len * edge_num)
    
    ----------
    Features of atom nodes:
    x[0]: Z, atomic number.
    x[1]: Group number, families of elements in the Periodic Table.
    x[2]: Period number, number of periods in the Periodic Table.
    x[3]: Electronegativity, electronegativity of atoms.
    x[4]: Covalent radius, covalent radius of single elements.
    x[5]: Valence electrons.
    x[6]: First ionization energy.
    x[7]: Electron affinity.
    x[8]: Block, s, p, d, f …
    x[9]: Atomic volume, volume of elemental atoms.
    
    ----------
    Adjacency matrix:
    Connected using Vesta:
    Vesta is a material structure visualization software, in which the searching algorithm for neighbor atoms is determined by molecular dynamics.
    `"New Method for Searching for Neighbors in Molecular Dynamics Computations"
    <https://doi.org/10.1016/0021-9991(73)90046-6>`_
    paper
    
    Connected using Voronoi:
    Voronoi is a search algorithm searching for neighbor atoms with coordination number.
    `"Crystal Space Analysis by means of Voronoi-Dirichlet Polyhedra"
    <https://doi.org/10.1107/S0108767395006799>`_
    paper
    
    ----------
    Features of bonds:
    edge_attr[0]: Atom distance, bonding of adjacent atoms.
    
    """
    #order_path = './elasiticity/bulk_modulus_index.csv'#Bulk Modulus
    order_path = './thermal_conduct/thermal_conductivity_index.csv'#Thermal Conductivity
    #data_path = "./data/data_voronoi/"
    data_path = "./data/thermo_conduct_struc_5k/"#Thermal Conductivity  # TODO: set your local data path
    poscar = Poscar.from_file(data_path + "mp-1000" + '/CONTCAR')#Thermal Conductivity
    structure = poscar.structure
    data = output_data_set(structure)