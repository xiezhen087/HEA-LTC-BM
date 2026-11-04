# -*- coding: utf-8 -*-
"""
Created on Mon May 19 17:56:53 2025

@author: Realunknown
"""
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
import matplotlib.pyplot as plt

def calculte_ltc(A ,M_ave,theta_D, delta, gruneisen, n, Temp):
    ltc = A*(M_ave*theta_D**3*delta)/(gruneisen**2*n**(2/3)*Temp)
    return ltc

def calculate_theta_D_B_p(B,poisson_ratio,M_ave):
    G = B*(3/2)*((1-2*poisson_ratio)/(1+poisson_ratio)) #Shear modulus for approximately isotropic materials (Anisotropy: 0.95~1.05) (unit: Pa)
    density_gcm3 = structure.density #g/cm^3
    rho = density_gcm3 * 1000 #kg/m^3
    v_l = np.sqrt((B + 4*G/3) / rho)
    v_t = np.sqrt(G / rho)
    v_m = (1/3 * (2/v_t**3 + 1/v_l**3))**(-1/3)
    M = M_ave/1000 #kg/mol

    # Calculate the Debye temperature
    hbar = 1.0545718e-34  # J·s
    kB = 1.380649e-23     # J/K
    NA = 6.02214076e23    # mol⁻¹
    pi = 3.1415926535897932384626433832795
    theta_D = (hbar / kB) * (6*pi**2 * NA * rho / M)**(1/3) * v_m
    return theta_D

"""
def calculate_f_sigma(poisson_ratio):
    f_sigma = ((3*((2/3)*((1+poisson_ratio)/(1-2*poisson_ratio)))**(3/2)+((1/3)*((1+poisson_ratio)/(1-poisson_ratio)))**(3/2))**(-1))**(1/3)
    return f_sigma
"""

def calculate_M_ave(structure):
    total_mass = 0.0
    for site in structure:
        element_symbol = site.species_string.split(',')[0].strip()
        element = Element(element_symbol)
        total_mass += element.atomic_mass
    avg_mass_g_per_mol = total_mass / len(structure)
    #avg_mass_g = avg_mass_g_per_mol / 6.02214076e23
    return avg_mass_g_per_mol

if __name__ == '__main__':
    #ltc_result_path = "./data/lattice_thermal_conduct_csv/"
    #ltc_result_path = "./data/qha/"
    #ltc_result_path = "./data/qha/"
    #ltc_result_path = "./data/result/"
    #ltc_result_path = "./data/qha/"
    #ltc_result_path = "./data/qha/"
    ltc_result_path = "./data/3Mo/"  # TODO: set your local data path
    #structure_path = "./data/struc_opt/"
    #structure_path = "./data/POSCAR_145_35.43/"
    #structure_path = "./data/qha/"
    #structure_path = "./data/result/"
    #structure_path = "./data/qha/"
    structure_path = "./data/3Mo/"  # TODO: set your local data path
    structure = Structure.from_file( structure_path + "POSCAR")
    
    #para_referenced    
    n = len(structure) #numbers of atoms in supercell
    Temp = 300 #K
    A = 3.1e-6 #constant
    kB = 1.380649E-23 #J/K
    hbar = 1.054571817E-34 #J⋅s
    V = structure.volume #Angstrom^3
    pi = 3.1415926535897932384626433832795
    
    #para_calculated
    #poisson_ratio = 0.271333333333333333
    #poisson_ratio = 0.308#from bulk_modulus calculation
    #poisson_ratio = 0.276
    #poisson_ratio = 0.28966666666666666666666666
    poisson_ratio = 0.32184174
    #gruneisen_temp = pd.read_csv( ltc_result_path + 'gruneisen_temp.csv').drop(index = 0)
    gruneisen_temp = pd.read_csv( ltc_result_path + 'gruneisen-temperature.csv').drop(index = 0)
    #bulk_modulus_temp = pd.read_csv(ltc_result_path + 'bulk_modulus_tem.csv').drop(index = 0) #GPa
    bulk_modulus_temp = pd.read_csv(ltc_result_path + 'bulk_modulus-tempareture.csv').drop(index = 0) #GPa
    #bulk_modulus = 352.972E9 #Pa
    #bulk_modulus = 348.58
    bulk_modulus = 194.408E9
    
    M_ave = calculate_M_ave(structure) #g/mol
    delta = (V/n)**(1/3) #Angstrom
    
    #bulk_modulus = bulk_modulus_temp["bulk_modulus"][30] * 10**9 #Pa
    theta_D = calculate_theta_D_B_p(bulk_modulus,poisson_ratio,M_ave)
    gruneisen = gruneisen_temp["gruneisen"]
    #ltc_temp = calculte_ltc(A ,M_ave,theta_D, delta, gruneisen, n, Temp)
    
    temp_list = []
    theta_D_list = []
    ltc_temp_list = []
    j = 1
    for i in range(len(bulk_modulus_temp)):
        temp_list.append(bulk_modulus_temp["temp"][j])
        theta_D_list.append(calculate_theta_D_B_p(bulk_modulus_temp["bulk_modulus"][j]* 10**9,poisson_ratio,M_ave))
        ltc_temp_list.append(calculte_ltc(A ,M_ave,theta_D_list[i], delta, gruneisen_temp["gruneisen"][j], n, temp_list[i]))
        j += 1
        
    #plot
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    
    #plt.plot(temp_list, ltc_temp_list, label='LTC')
    plt.plot(temp_list[0:59], ltc_temp_list[0:59], label='LTC')
    
    plt.title('LTC-TEMP')
    plt.xlabel('Temparature/K')
    plt.ylabel(r'Lattice Thermal Conductivity/W·$m^{-1}$·$K^{-1}$')
    
    plt.legend()
    plt.xlim(0,500)
    plt.ylim(0,30)
    
    plt.show()
    
    """
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.plot(temp_list, gruneisen_temp["gruneisen"], label='gruneisen')
    
    plt.title('Gruneisen-TEMP')
    plt.xlabel('Temparature')
    plt.ylabel('Gruneisen')
    
    plt.legend()
    
    plt.show()
    """