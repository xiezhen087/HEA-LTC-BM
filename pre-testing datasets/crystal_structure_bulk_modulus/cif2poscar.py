# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:23:30 2024

@author: Realunknown
"""
import os
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar

if __name__ == '__main__' :
    #elements_type = ["bi_alloy","ter_alloy","quater_alloy","qui_alloy"]
    elements_type = ["qui_alloy"]
    #structures_type = ["fcc","bcc","hcp"]
    structures_type = ["bcc_new"]
    structure_path = "D:/新建文件夹/经历：课题组事务/科研任务：机器学习/6.HEA组分和比例优化/searching/crystal_structure_lattice_thermal_conduct/"
    for element_type in elements_type:
        for structure_type in structures_type:
            path = structure_path + element_type + '/' + structure_type + '/'
            for root, dirs, files in os.walk(path):
                for name in dirs:
                    file_path = root + name + '/'
                    try:
                        parser = CifParser(file_path + "pos.cif")
                        structure = parser.get_structures()[0]
                        poscar = Poscar(structure)
                        poscar.write_file(file_path + "POSCAR")
                    except:
                        poscar_path = file_path + "POSCAR"
                        if not os.path.isfile(poscar_path):
                            print(file_path)