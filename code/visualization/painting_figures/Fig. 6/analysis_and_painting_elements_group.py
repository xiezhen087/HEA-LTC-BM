# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:21:11 2025

@author: Realunknown
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
import pandas as pd
import numpy as np
import shutil
num_valence_electrons =  {'Ac': 3, 'Ag': 11, 'Al': 3, 'Am': 9, 'Ar': 8, 'As': 5, 'At': 7, 'Au': 11, 'B': 3, 'Ba': 2, 'Be': 2, 'Bi': 5, 'Br': 7, 'C': 4, 'Ca': 2, 'Cd': 12, 'Ce': 4, 'Cl': 7, 'Cm': 10, 'Co': 9, 'Cr': 6, 'Cs': 1, 'Cu': 11, 'Dy': 12, 'Er': 14, 'Eu': 9, 'F': 7, 'Fe': 8, 'Fr': 1, 'Ga': 3, 'Gd': 10, 'Ge': 4, 'H': 1, 'He': 2, 'Hf': 4, 'Hg': 12, 'Ho': 13, 'I': 7, 'In': 3, 'Ir': 9, 'K': 1, 'Kr': 8, 'La': 3, 'Li': 1, 'Lu': 3, 'Mg': 2, 'Mn': 7, 'Mo': 6, 'N': 5, 'Na': 1, 'Nb': 5, 'Nd': 6, 'Ne': 8, 'Ni': 10, 'Np': 7, 'O': 6, 'Os': 8, 'P': 5, 'Pa': 5, 'Pb': 4, 'Pd': 10, 'Pm': 7, 'Po': 6, 'Pr': 5, 'Pt': 10, 'Pu': 8, 'Ra': 2, 'Rb': 1, 'Re': 7, 'Rh': 9, 'Rn': 8, 'Ru': 8, 'S': 6, 'Sb': 5, 'Sc': 3, 'Se': 6, 'Si': 4, 'Sm': 8, 'Sn': 4, 'Sr': 2, 'Ta': 5, 'Tb': 11, 'Tc': 7, 'Te': 6, 'Th': 4, 'Ti': 4, 'Tl': 3, 'Tm': 15, 'U': 6, 'V': 5, 'W': 6, 'Xe': 8, 'Y': 3, 'Yb': 16, 'Zn': 12, 'Zr': 4}
#plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20

delta_H_a_path = "./ΔHmin_a.csv"
delta_H_b_up_path = "./ΔHmin_b_up.csv"
delta_H_b_down_path = "./ΔHmin_b_down.csv"
delta_H_a = pd.read_csv(delta_H_a_path, header = 0,index_col = 0)
delta_H_b_up = pd.read_csv(delta_H_b_up_path, header = 0,index_col = 0, encoding='gbk')
delta_H_b_down = pd.read_csv(delta_H_b_down_path, header = 0,index_col = 0)
delta_H_a_number_row = [1,3,4,5,6,7,11,12,13,14,15,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,37,38,39,40,41,42,43,44,45,46]
delta_H_a_number_column = [46,47,48,49,50,51,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,90,92,94]
delta_H_b_number_up = [1,3,4,5,6,7,11,12,13,14,15,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,37,38,39,40,41,42,43,44,45]
delta_H_b_number_down = [47,48,49,50,51,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,90,92,94]

def get_delta_H_mix_AB(Z_i,Z_j):
    Z_max = max(Z_i,Z_j)
    Z_min = min(Z_i,Z_j)
    if Z_min in delta_H_a_number_row and Z_max in delta_H_a_number_column:
        delta_H_mix_AB = delta_H_a[str(Z_min)][Z_max]
    elif Z_min in  delta_H_b_number_up and Z_max in delta_H_b_number_up:
        delta_H_mix_AB = delta_H_b_up[str(Z_max)][Z_min]
    elif Z_min in  delta_H_b_number_down and Z_max in delta_H_b_number_down:
        delta_H_mix_AB = delta_H_b_down[str(Z_min)][Z_max]
    return delta_H_mix_AB

def calculate_delta_H_mix(structure):
    elements = [atom.specie.Z for atom in structure]
    element = list(set(elements))
    elements_ratio = [elements.count(ele)/(len(elements)) for ele in element]
    H_mix_ij_list = []
    for i in range(len(element) - 1):
        for j in range(i + 1,len(element)):
            Z_i = element[i]
            Z_j = element[j]
            c_i = elements_ratio[i]
            c_j = elements_ratio[j]
            delta_H_mix_AB = get_delta_H_mix_AB(Z_i,Z_j)
            delta_H_mix_AB = float(delta_H_mix_AB)
            """
            print(Element.from_Z(element[i]))
            print(Element.from_Z(element[j]))
            print(delta_H_mix_AB)
            print("***************************")
            """
            H_mix_ij = 4 * delta_H_mix_AB * c_i * c_j
            H_mix_ij_list.append(H_mix_ij)
    H_mix = sum(H_mix_ij_list)
    return H_mix

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

def calculate_delta(structure):
    delta = 1000
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

def cp_data(subspace):
    tgt_path = "./data/Fig. 6/"  # TODO: set your local data path
    src_path = "./data/formal/"  # TODO: set your local data path
    data_tgt_path = tgt_path + subspace + "/"
    
    if subspace == "bm_ternary_fcc":
        data_src_path = src_path + "250314_ternary_fcc_bulk_modulus"
    elif subspace == "bm_quinary_fcc":
        data_src_path = src_path + "250126_quinary_fcc_bulk_modulus"
    elif subspace == "bm_quinary_bcc":
        data_src_path = src_path + "250313_quinary_bcc_bulk_modulus"
    elif subspace == "ltc_quinary_fcc":
        data_src_path = src_path + "250401_quinary_fcc_thermal_conduct"
    elif subspace == "ltc_quinary_bcc":
        data_src_path = src_path + "250618_quinary_bcc_thermal_conduct"
    elif subspace == "ltc_quinary_added_251026_bcc":
        data_src_path = src_path + "supplementary_fcc_bcc_ltc_single_objective/251026_quinary_bcc_thermal_conduct"  # TODO: set your local data path
    elif subspace == "ltc_quinary_added_251021_fcc":
        data_src_path = src_path + "supplementary_fcc_bcc_ltc_single_objective/251021_quinary_fcc_thermal_conduct"  # TODO: set your local data path
    elif subspace == "ltc_quinary_sum_250430_fcc":
        data_src_path_m = [src_path + "supplementary_fcc_bcc_ltc_single_objective/251021_quinary_fcc_thermal_conduct", src_path + "250401_quinary_fcc_thermal_conduct"]  # TODO: set your local data path
    elif subspace == "ltc_quinary_added_251014_bcc":
        data_src_path = src_path + "supplementary_fcc_bcc_ltc_single_objective/251014_quinary_bcc_thermal_conduct"  # TODO: set your local data path
    
    if subspace == "ltc_quinary_sum_250430_fcc":
        data_files_list_1 = os.listdir(data_src_path_m[0])
        data_files_list_2 = os.listdir(data_src_path_m[1])
        data_files_list = data_files_list_1 + data_files_list_2
    else:
        data_files_list = os.listdir(data_src_path)
    null = 0
    for data_files in data_files_list:
        if subspace == "ltc_quinary_sum_250430_fcc":
            if "251021" in data_files:
                data_file_list = os.listdir(data_src_path_m[0] + "/" + data_files + "/")
            elif ("250506" in data_files) or ("250401" in data_files):
                data_file_list = os.listdir(data_src_path_m[1] + "/" + data_files + "/")
        else:
            data_file_list = os.listdir(data_src_path + "/" + data_files + "/")
        for data_file in data_file_list:
            if data_file[0] == "y":
                continue
            name = data_file.split("_")
            if len(name) == 4:
                value = float(name[-2])
            elif len(name) == 3:
                value = float(name[-1])
            if value < 0:
                continue
            else:
                #print(data_src_path + "/" + data_files + "/" + data_file)
                if subspace == "ltc_quinary_sum_250430_fcc":
                    if "251021" in data_files:
                        data_src_path = data_src_path_m[0]
                    elif ("250506" in data_files) or ("250401" in data_files):
                        data_src_path = data_src_path_m[1]
                poscar = Poscar.from_file(data_src_path + "/" + data_files + "/" + data_file,'r')
                structure = poscar.structure
                delta = calculate_delta(structure)
                vec,_,_ = calculate_vec(structure)
                H_mix = calculate_delta_H_mix(structure)
                if (delta < 0.045) and (H_mix > -22 and H_mix < 5):
                    if subspace[-3:] == "fcc" and vec > 8.0:
                        shutil.copy(data_src_path + "/" + data_files + "/" + data_file, data_tgt_path  + "/" + data_file)
                    elif subspace[-3:] == "bcc" and vec < 6.8:
                        shutil.copy(data_src_path + "/" + data_files + "/" + data_file, data_tgt_path  + "/" + data_file)
                    else:
                        null += 1
                else:
                    null += 1
    print(null)

def value_mapping(value, subspace):
    if subspace == "bm_ternary_fcc":
        value_mapped = (value + 2.60)/0.16 
    elif subspace == "bm_quinary_fcc":
        value_mapped = (value + 9.04)/0.25
    elif subspace == "bm_quinary_bcc":
        value_mapped = (value + 2.38)/0.13
    elif subspace == "ltc_quinary_fcc" or subspace == "ltc_quinary_added_251021_fcc" or subspace == "ltc_quinary_sum_250430_fcc":
        value_mapped = (value + 7.75)/5.88
    elif subspace == "ltc_quinary_bcc" or subspace == "ltc_quinary_added_251026_bcc" or subspace == "ltc_quinary_added_251014_bcc":
        value_mapped = (value + 5.03)/1.28
    return value_mapped

if __name__ == '__main__' :
    #subspaces = ["bm_ternary_fcc","bm_quinary_fcc","bm_quinary_bcc","ltc_quinary_fcc","ltc_quinary_fcc"]
    #subspaces = ["ltc_quinary_added_251026_bcc"]
    #subspaces = ["ltc_quinary_added_251021_fcc"]
    #subspaces = ["ltc_quinary_added_251014_bcc"]
    #subspaces = ["ltc_quinary_fcc"]
    #subspaces = ["bm_quinary_fcc","bm_quinary_bcc","ltc_quinary_added_251026_bcc"]
    #subspaces = ["bm_quinary_bcc"]
    subspaces = ["bm_quinary_fcc","bm_quinary_bcc","ltc_quinary_added_251026_bcc","ltc_quinary_sum_250430_fcc"]
    for subspace in subspaces:
        #cp_data(subspace)
        file_path = "./" + subspace + "/"
        file_name_list = os.listdir(file_path)
        data_point_list = []
        for file_name in file_name_list:
            data_point = {}
            poscar = Poscar.from_file(file_path + file_name,'r')
            structure = poscar.structure
            elements = [atom.specie.symbol for atom in structure]
            elements_label = list(set(elements))
            name = file_name.split("_")
            if len(name) == 4:
                value = float(name[-2])
            elif len(name) == 3:
                value = float(name[-1])
            data_point["elements"] = elements_label
            value = value_mapping(value, subspace)
            data_point["value"] = value
            data_point_list.append(data_point)
        
        #Element counts
        if "ter" in subspace:
            element_num = 3
        elif "qui" in subspace:
            element_num = 5
            
        total_elements = []
        for data_point in data_point_list:
            total_elements = total_elements + data_point["elements"]
        series = pd.Series(total_elements)
        element_count = series.value_counts()
        element_label = list(set(total_elements))
        
        df_elements = []
        df_value = []
        df_label = []
        for data_point in data_point_list:
            df_elements = df_elements + data_point["elements"]
            df_value = df_value + [data_point["value"]] * element_num
            
            if "ter" in subspace:
                label =  data_point["elements"][0] + "-"+ data_point["elements"][1] + "-"+ data_point["elements"][2]
            elif "qui" in subspace:
                label =  data_point["elements"][0] + "-"+ data_point["elements"][1] + "-"+ data_point["elements"][2] + "-"+ data_point["elements"][3] + "-"+data_point["elements"][4]
            
            df_label = df_label + [label]*element_num
            
        df_data = {"elements":df_elements, "value":df_value,"label":df_label}
        df = pd.DataFrame(df_data)
        plt.figure(dpi=600)
        g = sns.catplot(x="elements", y="value",kind="swarm", hue = "label",height=6,aspect=3, data=df)
        g._legend.remove()
        
        # ====================== Fix: solid black legend border (compatible with all versions) ======================
        if "ter" in subspace:
            leg = plt.legend(fontsize=18,bbox_to_anchor=(1.05, -0.05),ncol = 8,edgecolor="black")
        elif "qui" in subspace:
            leg = plt.legend(fontsize=18,bbox_to_anchor=(1.04, -0.15),ncol = 6,edgecolor="black")
        if subspace == "bm_quinary_bcc":
            leg = plt.legend(fontsize=18,bbox_to_anchor=(1.04, -0.3),ncol = 6,edgecolor="black")
        
        # Compatible approach: set the legend border linewidth
        leg.get_frame().set_linewidth(1.5)
            
        plt.xlabel("Elements")
        if subspace[:2] == "bm":
            #plt.ylabel("Predicted Bulk Modulus / GPa")
            plt.ylabel("Predicted Bulk Modulus [GPa]")
        elif subspace[:3] == "ltc":
            #plt.ylabel("Predicted Lattice Thermal Conductivity / $\mathrm{W \cdot m^{-1} \cdot K^{-1}}$")
            plt.ylabel("Predicted Lattice Thermal Conductivity [$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]")
            
        if subspace == "bm_ternary_fcc":
            plt.title("Predicted Bulk Modulus in Ternary FCC")
        elif subspace == "bm_quinary_fcc":
            plt.title("Predicted Bulk Modulus in Quinary FCC")
        elif subspace == "bm_quinary_bcc":
            plt.title("Predicted Bulk Modulus in Quinary BCC", pad=20)
        elif subspace == "ltc_quinary_fcc" or subspace == "ltc_quinary_added_251021_fcc" or subspace == "ltc_quinary_sum_250430_fcc":
            plt.title("Predicted Lattice Thermal Conductivity in Quinary FCC",pad = 50)
        elif subspace == "ltc_quinary_bcc" or subspace == "ltc_quinary_added_251026_bcc":
            plt.title("Predicted Lattice Thermal Conductivity in Quinary BCC",pad = 50)
        
        # ====================== Restore top + right spines ======================
        ax = plt.gca()
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        
        #plt.tight_layout()
        #plt.savefig( subspace + '.png', dpi=600, bbox_inches='tight')
        plt.show()