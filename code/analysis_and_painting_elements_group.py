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

if __name__ == '__main__' :
    file_path = "../searching/log/final/multi_goal/final_250327_nsga_quinary_fcc_multi_goal/below_60/efficient_data_equiatoms/"
    file_name_list = os.listdir(file_path)
    data_point_list = []
    for file_name in file_name_list:
        poscar_file_list = os.listdir(file_path + file_name)
        for poscar_name in poscar_file_list:
            data_point = {}
            poscar = Poscar.from_file(file_path + file_name + "/" + poscar_name,'r')
            structure = poscar.structure
            elements = [atom.specie.symbol for atom in structure]
            elements_label = list(set(elements))
            name = poscar_name.split("_")
            if len(name) == 4:
                value = float(name[-2])
            elif len(name) == 3:
                value = float(name[-1])
            data_point["elements"] = elements_label
            data_point["value"] = value
            data_point_list.append(data_point)
    
    #Element counts
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
        label =  data_point["elements"][0] + "-"+ data_point["elements"][1] + "-"+ data_point["elements"][2] + "-"+ data_point["elements"][3] #+ "-"+data_point["elements"][4]
        #label =  data_point["elements"][0] + "-"+ data_point["elements"][1] + "-"+ data_point["elements"][2]
        df_label = df_label + [label]*element_num
    df_data = {"elements":df_elements, "value":df_value,"label":df_label}
    df = pd.DataFrame(df_data)
    plt.figure(dpi=300)
    g = sns.catplot(x="elements", y="value",kind="swarm", hue = "label",height=6,aspect=3, data=df)
    g._legend.remove()
    plt.legend(bbox_to_anchor=(1.0, -0.08),ncol = 11)
    plt.show()