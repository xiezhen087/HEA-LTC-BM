# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:36:51 2025

@author: Realunknown
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pymatgen.io.vasp import Poscar
import pandas as pd

#Old color
"""
element_color = {"Ti":"#d7dee7", "V":"#c9d3df", "Cr": "#bcc8d7" ,"Mn": "#aebccf", "Fe" : "#a1b1c7", "Co": "#93a6bf", "Ni": "#869bb7", "Cu": "#7890af", "Zn": "#6c829e",
    "W": "#efdfd6", "Ta": "#ead4c8", "Mo": "#e5caba", "Nb": "#e0bfac", "Zr" : "#dbb49e", "Hf" : "#d5a991", "Re":"#d09f83",
    "Al": "#ddecdb", "Mg": "#d1e5cf", "Be" : "#c6dfc3", "K": "#bad9b7", "Na" : "#afd2ab", "Sr" : "#a3cc9f", "Ba": "#98c593", "Ca": "#8cbf87", "Li":"#7eac7a",
    "Sn": "#c79994", "In":"#bf8a85", "Bi": "#b77c75", "Pb":"#af6d66",
    "Au": "#eae3e9", "Ag": "#e3dae1", "Pt" : "#dcd1da", "Pd":"#d4c8d3", "Ir": "#cdbfcb", "Os": "#c6b5c4", "Rh": "#bfacbc", "Ru":"#b8a3b5",
    "Sc": "#e8e8e8", "Y": "#e0e1e0", "La": "#d8d9d9", "Ce": "#d0d1d1", "Pr": "#c8cac9", "Nd": "#c1c2c1", "Pm":"#b9bbba", "Sm": "#b1b3b2", "Eu": "#9fa1a0", "Gd" : "#8e8f8e", "Tb": "#7c7d7d",
    "Dy": "#6a6b6b", "Ho": "#595a59", "Er" : "#474847", "Tm":"#353635", "Yb": "#f7f7f7", "Lu": "#eff0f0",
    "Tc": "#000000"
    }
"""

#New color
element_color = {"Ti":"#bcd6e9", "V":"#a5c9e1", "Cr": "#8fbbda" ,"Mn": "#79add2", "Fe" : "#62a0cb", "Co": "#4c92c3", "Ni": "#3585bc", "Cu": "#1f77b4", "Zn": "#1c6ba2",
    "W": "#ffb26e", "Ta": "#ffa556", "Mo": "#ff993e", "Nb": "#ff8c26", "Zr" : "#ff7f0e", "Hf" : "#e6720d", "Re":"#cc660b",
    "Al": "#d5ecd5", "Mg": "#c0e3c0", "Be" : "#abd9ab", "K": "#96d096", "Na" : "#80c680", "Sr" : "#6bbd6b", "Ba": "#56b356", "Ca": "#41aa41", "Li":"#2ca02c",
    "Sn": "#c79994", "In":"#bf8a85", "Bi": "#b77c75", "Pb":"#af6d66",
    "Au": "#dfd1eb", "Ag": "#d4c2e5", "Pt" : "#cab3de", "Pd":"#bfa4d7", "Ir": "#b495d1", "Os": "#a985ca", "Rh": "#9f76c4", "Ru":"#9467bd",
    "Sc": "#ddccc9", "Y": "#d1bbb7", "La": "#c6aba5", "Ce": "#ba9a93", "Pr": "#af8981", "Nd": "#a3786f", "Pm":"#98675d", "Sm": "#8c564b", "Eu": "#7e4d44", "Gd" : "#70453c", "Tb": "#623c35",
    "Dy": "#54342d", "Ho": "#462b26", "Er" : "#38221e","Tm" : "#2a1a17", "Yb": "#1c110f", "Lu": "#e8dddb",
    "Tc": "#000000"
    }

#element_color_fcc = {"W": "#636e72", "Pd" :"#b2bec3", "Cr": "#fdcb6e", "Ti": "#fd79a8", "Rh": "#0984e3", "Os": "#00cec9", "Re":"#a29bfe", "Ta": "#d63031", "Ru": "#00b894", "Pt":"#e17055", "Ir":"#74b9ff"}

def remove_duplicate(structure_file_list):
    structure_file_list_new = []
    element_label = []
    for structure_file in structure_file_list:
        stru_list = structure_file.split("_")
        ele = stru_list[0] + "-" + stru_list[1]
        if ele not in element_label:
            element_label.append(ele)
            structure_file_list_new.append(structure_file)
    return structure_file_list_new

if __name__ == '__main__' :
    structure_path = "./data/Hmix_meets/"  # TODO: set your local data path
    structure_file_list = os.listdir(structure_path)
    structure_file_list = remove_duplicate(structure_file_list)
    plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})
    
    element_list = []
    ratio_list = []
    ratio_add_up_list = []
    bm_function_list = []
    ltc_function_list = []
    for structure_file in structure_file_list:
        poscar = Poscar.from_file(structure_path + structure_file)
        structure = poscar.structure
        
        #elements_ratio 
        elements = [atom.specie.symbol for atom in structure]
        element_type = list(set(elements))
        elements_ratio = [elements.count(ele)/len(elements) for ele in element_type]
        ratio_list.append(elements_ratio)
        ratio_add_up = [elements_ratio[0], sum(elements_ratio[0:2]), sum(elements_ratio[0:3]), sum(elements_ratio[0:4])]
        
        pro = structure_file.split("_")
        ltc_function_list.append(float(pro[-1]))
        bm_function_list.append(float(pro[-2]))
        
        element_list.append(element_type)
        ratio_add_up_list.append(ratio_add_up)
        
        
    fig, ax = plt.subplots(figsize=(10,10),dpi = 600) 
    
    i=0 
    while i<len(bm_function_list): 
        x = [0] + np.cos(np.linspace(0, 2 * np.pi * ratio_add_up_list[i][0], 30)).tolist() 
        y = [0] + np.sin(np.linspace(0, 2 * np.pi *ratio_add_up_list[i][0], 30)).tolist() 
        xy1 = list(zip(x, y)) 
    
        x = [0] + np.cos(np.linspace(2 * np.pi * ratio_add_up_list[i][0], 2 * np.pi * ratio_add_up_list[i][1], 30)).tolist() 
        y = [0] + np.sin(np.linspace(2 * np.pi * ratio_add_up_list[i][0], 2 * np.pi * ratio_add_up_list[i][1], 30)).tolist() 
        xy2 = list(zip(x, y)) 
    
        x = [0] + np.cos(np.linspace(2 * np.pi * ratio_add_up_list[i][1], 2 * np.pi* ratio_add_up_list[i][2], 30)).tolist() 
        y = [0] + np.sin(np.linspace(2 * np.pi * ratio_add_up_list[i][1], 2 * np.pi* ratio_add_up_list[i][2], 30)).tolist() 
    
        xy3 = list(zip(x, y)) 
    
        x = [0] + np.cos(np.linspace(2 * np.pi * ratio_add_up_list[i][2], 2 * np.pi*1, 30)).tolist() 
        y = [0] + np.sin(np.linspace(2 * np.pi * ratio_add_up_list[i][2], 2 * np.pi*1, 30)).tolist() 
    
        xy4 = list(zip(x, y)) 
        
        x = [0] + np.cos(np.linspace(2 * np.pi * ratio_add_up_list[i][3], 2 * np.pi*1, 30)).tolist() 
        y = [0] + np.sin(np.linspace(2 * np.pi * ratio_add_up_list[i][3], 2 * np.pi*1, 30)).tolist() 
    
        xy5 = list(zip(x, y)) 
        
        xy=[xy1,xy2,xy3,xy4,xy5] 
        c=[element_color[element_list[i][0]],element_color[element_list[i][1]],element_color[element_list[i][2]],element_color[element_list[i][3]],element_color[element_list[i][4]]]
        for j in range(5): 
            ax.scatter(bm_function_list[i], ltc_function_list[i], marker=(xy[j]), s=2000,facecolor=c[j]) 
        i=i+1 
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    
    plt.xlabel("Bulk Modulus Index")
    plt.ylabel("Lattice Thermal Conductivity Index")
    plt.title("The correlation between quinary BCC alloy composition and properties",pad = 30)
    plt.show()
    
    system_choose = []
    for i in range(len(element_list)):
        system = []
        system.append(element_list[i])
        system.append(bm_function_list[i])
        system.append(ltc_function_list[i])
        system_choose.append(system)
        
    result_list = []
    for i in range(len(ratio_list)):
        result = []
        composition = system_choose[i][0][0] + str(int(108*ratio_list[i][0])) + "-" + system_choose[i][0][1] + str(int(108*ratio_list[i][1])) + "-" + system_choose[i][0][2] + str(int(108*ratio_list[i][2]))+ "-" + system_choose[i][0][3] + str(int(108*ratio_list[i][3]))+ "-" + system_choose[i][0][4] + str(int(108*ratio_list[i][4]))
        poscar_name = structure_file_list[i]
        bm_pred = system_choose[i][1]
        ltc_pred = system_choose[i][2]
        result.append(composition)
        result.append(poscar_name)
        result.append(bm_pred)
        result.append(ltc_pred)
        result_list.append(result)
        
    df = pd.DataFrame(data = result_list)
    df.to_csv('FCC.csv', encoding='utf-8')