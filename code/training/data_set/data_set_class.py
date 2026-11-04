# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:41:03 2024

@author: PC
"""
import torch
import os
from pymatgen.core.periodic_table import Element

metal_elements = ["Li","Na","K","Rb","Cs","Fr","Be","Mg","Ca","Sr","Ba","Ra","Sc","Y","Ti","Zr","Hf","Rf","V","Nb","Ta","Db","Cr","Mo","W","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh","Ir","Mt","Ni","Pb","Fl","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Ge","Sn","Sb","Bi","Mc","Po","Lv","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"]
metal_elements_no_radiation = ["Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba","Sc","Y","Ti","Zr","Hf","V","Nb","Ta","Cr","Mo","W","Mn","Tc","Re","Fe","Ru","Os","Co","Rh","Ir","Ni","Pb","Pt","Cu","Ag","Au","Zn","Cd","Hg","Al","Ga","In","Tl","Ge","Sn","Sb","Bi","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"]

if __name__ == '__main__' :
    data_path = './data/Thermo_conduct/'#class  # TODO: set your local data path
    new_data_set = []
    Data_set = torch.load(os.path.join(data_path + 'cgcnn-voronoi-Phonon_thermal_conduct.pt'))
    i = 0
    for data in Data_set:
        if i %10000 == 0:
            print(i)
        atomic_number = []
        for ele in data.x:
            atomic_number.append(ele[0])
        element = [Element.from_Z(an).symbol  for an in atomic_number]
        element = list(set(element))
        if_metal = []
        for ele in element:
            if ele in metal_elements:
                if_metal.append(1)
            else:
                if_metal.append(0)
#        if ("Fe" in element) or ("Ni" in element) or ("Cr" in element) or ("Co" in element):
        if sum(if_metal) == len(if_metal):
            new_data_set.append(data)
        i += 1
#    torch.save(new_data_set, data_path + "cgcnn-voronoi-Formation_energy-metal-1w.pt")