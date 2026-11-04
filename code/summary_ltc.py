# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:20:45 2025

@author: Realunknown
"""

import os
from shutil import copyfile
import shutil

structure_data = "./data/structure_and_data_selected/"  # TODO: set your local data path
tgt_path = "./data/summary_ltc_structure_selected/"  # TODO: set your local data path

if __name__ == '__main__' :
    series_file_name_list = os.listdir(structure_data)
    for series_name in series_file_name_list:
        generation_list = os.listdir(structure_data + "/" + series_name + "/")
        os.mkdir(tgt_path + "/" + series_name + "/")
        #element_type_list = []
        for generation in generation_list:
            single_file_name_list = os.listdir(structure_data + "/" + series_name + "/" + generation + "/")
            for single_file_name in single_file_name_list:
                #information = single_file_name.split("_")
                #if information[1] not in element_type_list:
                #element_type_list.append(information[1])
                source = structure_data + "/" + series_name + "/" + generation + "/" + single_file_name
                target = tgt_path + "/" + series_name + "/" + single_file_name
                shutil.copy(source, target)