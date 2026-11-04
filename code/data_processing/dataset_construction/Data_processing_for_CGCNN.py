# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 07:54:53 2022

@author: Realunknown
"""
import os
from pymatgen.io.vasp import Poscar
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
import torch
from torch_geometric.data import Data
from scipy.constants import Avogadro
import math
from torch.nn.functional import one_hot as to_one_hot
import numpy as np
import csv

num_valence_electrons =  {'Ac': 3, 'Ag': 11, 'Al': 3, 'Am': 9, 'Ar': 8, 'As': 5, 'At': 7, 'Au': 11, 'B': 3, 'Ba': 2, 'Be': 2, 'Bi': 5, 'Br': 7, 'C': 4, 'Ca': 2, 'Cd': 12, 'Ce': 4, 'Cl': 7, 'Cm': 10, 'Co': 9, 'Cr': 6, 'Cs': 1, 'Cu': 11, 'Dy': 12, 'Er': 14, 'Eu': 9, 'F': 7, 'Fe': 8, 'Fr': 1, 'Ga': 3, 'Gd': 10, 'Ge': 4, 'H': 1, 'He': 2, 'Hf': 4, 'Hg': 12, 'Ho': 13, 'I': 7, 'In': 3, 'Ir': 9, 'K': 1, 'Kr': 8, 'La': 3, 'Li': 1, 'Lu': 3, 'Mg': 2, 'Mn': 7, 'Mo': 6, 'N': 5, 'Na': 1, 'Nb': 5, 'Nd': 6, 'Ne': 8, 'Ni': 10, 'Np': 7, 'O': 6, 'Os': 8, 'P': 5, 'Pa': 5, 'Pb': 4, 'Pd': 10, 'Pm': 7, 'Po': 6, 'Pr': 5, 'Pt': 10, 'Pu': 8, 'Ra': 2, 'Rb': 1, 'Re': 7, 'Rh': 9, 'Rn': 8, 'Ru': 8, 'S': 6, 'Sb': 5, 'Sc': 3, 'Se': 6, 'Si': 4, 'Sm': 8, 'Sn': 4, 'Sr': 2, 'Ta': 5, 'Tb': 11, 'Tc': 7, 'Te': 6, 'Th': 4, 'Ti': 4, 'Tl': 3, 'Tm': 15, 'U': 6, 'V': 5, 'W': 6, 'Xe': 8, 'Y': 3, 'Yb': 16, 'Zn': 12, 'Zr': 4}

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

def get_x_ordering_para(para):
    r"""
    This function is used to get the ordering parameters.
    
    """
    alpha = para[0]
#    alpha_para = to_one_hot(torch.tensor(alpha),num_classes = 278)
    alpha_para = math.log(alpha + 1,20)
    beta = para[1]
#    beta_para = to_one_hot(torch.tensor(beta),num_classes = 302)
    beta_para = math.log(alpha + 1, 30)
    gamma = para[2]
#    gamma_para = to_one_hot(torch.tensor(gamma),num_classes = 9)
    gamma_para = math.log(alpha + 1, 10)
    omega = para[3]
    omega_para = to_one_hot(torch.tensor(omega),num_classes = 6)
    return torch.tensor([alpha_para]),torch.tensor([beta_para]),torch.tensor([gamma_para]),omega_para

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
        #alpha_para,beta_para,gamma_para,omega_para = get_x_ordering_para(ordering_paras[i])
        #atom_fea.append(get_x_mass(structure[i]))
        #atom_fea.append(alpha_para)
        #atom_fea.append(beta_para)
        #atom_fea.append(gamma_para)
        #atom_fea.append(omega_para)
        #atom_fea = torch.cat(atom_fea)
        x.append(atom_fea)
    return torch.Tensor(x)

def get_edge_index(file_path ,file_name ,search_method):
    r"""
    This function imports the adjacency matrix of the structure.
    """
    if (search_method == "vesta") or (search_method == "vesta_phonon"):
        edge_index = torch.load(os.path.join(file_path + '/', file_name + '-edge_index.pt'))
    elif (search_method == "voronoi") or (search_method == "voronoi_phonon") or (search_method == "voronoi_elasticity"):
        edge_index = torch.tensor(torch.load(os.path.join(file_path + '/', file_name + '-edge_index_from_voronoi.pt')))
    return edge_index

def get_edge_attr(file_path ,file_name ,search_method):
    r"""
    This function imports chemical bond long data from files.
    """
    edge_attr = []
    radius = 8
    dmin = 0
    dmax = radius
    step = 0.2
    edge_filter = np.arange(dmin, dmax+step, step)
    if (search_method == "vesta") or (search_method == "vesta_phonon"):
        bond_length = torch.load(os.path.join(file_path + '/', file_name + '-edge_lenth.pt'))
    elif (search_method == "voronoi") or (search_method == "voronoi_phonon") or (search_method == "voronoi_elasticity"):
        bond_length = torch.load(os.path.join(file_path + '/', file_name + '-edge_length_from_voronoi.pt'))
    #Chemical bond characteristics are divided into ten categories according to different lengths
    bond_length = np.array(bond_length)
    edge_attr = np.exp(-(bond_length[..., np.newaxis] - edge_filter)**2 /
                      step**2)
    return torch.tensor(edge_attr)

def get_label(labels_path, file_path, file_name , labels_tag ,classification):
    r"""
    """
    if classification == True:
        data_before = torch.load(os.path.join(file_path + '/', file_name + '.pt'))
        y = data_before.y
    else:
        try:
            if labels_tag == "Band_gap":
                y = torch.load(os.path.join(labels_path + '/' + file_name[:-9], file_name[:-9] + '-20230409-properties.pt'))[1]
            elif labels_tag == "Formation_energy":
                #y = torch.load(os.path.join(labels_path + '/' + file_name[:-9], file_name[:-9] + '-20230409-properties.pt'))[2]
                y = torch.load(os.path.join(labels_path + '/' + file_name, file_name + '-20230409-properties.pt'))[2]
            elif labels_tag == "Fermi_energy":
                y = torch.load(os.path.join(labels_path + '/' + file_name[:-9], file_name[:-9] + '-20230409-properties.pt'))[3]
            elif labels_tag == "Total_energy":
                y = torch.load(os.path.join(labels_path  + file_name, file_name + '-20230409-total_energy.pt'))[1]
            elif labels_tag == "Phonon_free_energy":
                y = torch.load(os.path.join(file_path , file_name[:-9] + '-phonon_prop.pt'))[0]
            elif labels_tag == "Entropy":
                y = torch.load(os.path.join(file_path, file_name[:-9] + '-phonon_prop.pt'))[1]
            elif labels_tag == "Phonon_thermal_capacity":
                y = torch.load(os.path.join(file_path, file_name[:-9] + '-phonon_prop.pt'))[2]
            elif labels_tag == "elasticity_poisson_ratio":
                y = torch.load(os.path.join(labels_path + '/' + file_name, file_name + '-20230409-elasticity.pt'))[1]["poisson_ratio"]
            elif labels_tag == "elastic_anisotropy":
                y = torch.load(os.path.join(labels_path + '/' + file_name, file_name + '-20230409-elasticity.pt'))[1]["elastic_anisotropy"]
            elif labels_tag == "G_Voigt_Reuss_Hill":
                y = torch.load(os.path.join(labels_path + '/' + file_name, file_name + '-20230409-elasticity.pt'))[1]["G_Voigt_Reuss_Hill"]#shear modulus
            elif labels_tag == "K_Voigt_Reuss_Hill":
                y = torch.load(os.path.join(labels_path + '/' + file_name, file_name + '-20230409-elasticity.pt'))[1]["K_Voigt_Reuss_Hill"]#bulk modulus
            elif labels_tag == "E_Voigt_Reuss_Hill":
                pr = torch.load(os.path.join(labels_path + '/' + file_name, file_name + '-20230409-elasticity.pt'))[1]["poisson_ratio"]
                g = torch.load(os.path.join(labels_path + '/' + file_name, file_name + '-20230409-elasticity.pt'))[1]["G_Voigt_Reuss_Hill"]
                y = 2*g*(1 + pr)#Young's modulus
            elif labels_tag == "Phonon_thermal_conduct":
                y = torch.load(os.path.join(labels_path + file_name, file_name + '-tc.pt'))[-1]
        except:
            y = -1
        y = torch.tensor(y)
    return y

def get_structures(file_path):
    r"""
    In this function, the pymatgen module is used to import the target POSCAR structure file and convert it into a structure object.
    
    """
    #poscar = Poscar.from_file(file_path + "POSCAR",'r')
    poscar = Poscar.from_file(file_path + "CONTCAR",'r')#Thermo_conduct.
    structure = poscar.structure
    return structure

def delete_ima_freq_data(files_list):
    no_ima_freq_path = "./data/normal_mp_id.csv"  # TODO: set your local data path
    files_name_list = []
    files_list_new = []
    with open(no_ima_freq_path) as file:
        csv_reader = csv.reader(file)
        list_of_csv = list(csv_reader)
    for ele in list_of_csv:
        files_name_list.append(ele[0])
    for files_name in files_name_list:
        files_list_new.append('mp-' + files_name + '-20180417')
    return files_list_new

def save(target_path ,data ,search_method,labels_tag): 
    torch.save(data, os.path.join(target_path + "cgcnn" + "-" + search_method + "-" + labels_tag +'.pt'))
    return 0

if __name__ == '__main__' :
    r"""
    This script is used to construct the graph data structure of the CGCNN model. 
    Part of the data was imported from an existing database, such as POSCAR files, edge_index of vesta, edge_index of voronoi, labels, edge_attr.
    
    ----------
    Contents in Data:
    x: The features of atom nodes.(Tensor: N * atom_fea_len)
    edge_index: Adjacency matrix represented by COO form (index of source atoms and index of target atoms,Tensor: 2 * edge_num)
    edge_attr: The features of bonds.(Tensor: bond_fea_len * edge_num)
    y: The labels(Tensor: 1 * 1 (or 2 for classification))
    
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
    
    ----------
    Labels:
    Classification (classification == True):
    Virtual_frequency_exist: Yes or No (1.0 or 0.0).
    
    Regression (classification == False):
    Formation_energy: a physical quantity.
    Band_gap: the energy difference between the top and bottom of the material price band.
    Fermi_energy: a physical quantity.
    Total_energy: the total energy of materials.
    
    Phonon_entropy: a physical quantity related to phonons.
    Phonon_enthalpy: a physical quantity related to phonons.
    Phonon_thermal_capacity: a physical quantity related to phonons.
    
    
    """
#    structural_files_path = "./data/Configuration_new_edgeindex/" #Thermo Capaci.
#    structural_files_path = "./data/data_voronoi/" #Energy
    structural_files_path = "./data/thermo_conduct_struc_5k/" #Thermo conduct.  # TODO: set your local data path
#    elasticity_path = "./data/elasticity_1w/"#Elasticity

#    files_list = os.listdir(elasticity_path)
    files_list = os.listdir(structural_files_path)
    
#    labels_path = "./data/Configuration_new_edgeindex/" #Thermo Capaci.
#    labels_path = "./data/properties_13w/" #Energy
#    labels_path = "./data/total_energy_13w/" #Energy
#    labels_path = "./data/elasticity_1w/" #Elasticity
    labels_path = "./data/thermo_conduct_5k/" #Thermo conduct.  # TODO: set your local data path

    files_list = os.listdir(labels_path)
#    files_list = delete_ima_freq_data(files_list)#Thermo Capaci.

#    target_path = "./data/Thermo_capaci/"
#    target_path = "./data/Energy/"
    target_path = "./data/Thermo_conduct/"  # TODO: set your local data path
#    target_path = "./data/Elasticity/"

    classification = False
    data_set = []
    files_list = files_list[:126181] + files_list[126182:]
    ini = 0
    #ini = 35054
    fin = len(files_list)
#    for file in files_list:
    for i in range(ini,fin):
        file_index = files_list.index(files_list[i])
        print(file_index)
#        if (file_index%10000) == 2:
            #time.sleep(30)
#        print(files_list.index(file))

        #structure = get_structures(structural_files_path + files_list[i] + '/')
        structure = get_structures(structural_files_path + files_list[i][5:] + '/')
        
        #x = get_atom_features(structure,structural_files_path + file + '/',file)
        x = get_atom_features(structure)
        
        #edge_index = get_edge_index(structural_files_path + files_list[i] + '/',files_list[i] , search_method = "voronoi")
        edge_index = get_edge_index(structural_files_path + files_list[i][5:] + '/',files_list[i][5:] , search_method = "voronoi")

        #edge_attr = get_edge_attr(structural_files_path + files_list[i] + '/', files_list[i], search_method = "voronoi")
        edge_attr = get_edge_attr(structural_files_path + files_list[i][5:] + '/', files_list[i][5:], search_method = "voronoi")

        y = get_label(labels_path,structural_files_path + files_list[i] + '/',files_list[i] ,labels_tag = "Phonon_thermal_conduct",classification = False)
        
        #if y < 0:#The range of values for physical quantities.
        #if y > 0:
        #    continue
        
        #if classification == False:
            #if float(y) == -1.0: # If there is no data for this mp-ID, skip
                #continue
        data = Data(x = x,edge_index = edge_index,edge_attr = edge_attr,y = y)
        data_set.append(data)
    #save(target_path ,data_set ,search_method = "voronoi",labels_tag = "Phonon_thermal_conduct")
        #break