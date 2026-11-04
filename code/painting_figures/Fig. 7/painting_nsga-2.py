# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 22:35:27 2025

@author: Realunknown
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.rcParams.update({
    #'font.family': 'Times New Roman',
    'font.family': 'Arial',
    'font.size': 24,
})
    data_path = "./data/Fig. 7/"  # TODO: set your local data path
    index = [1, 6, 11, 16, 21, 26]
    #index = [26]
    for i in index:
        file_name = "250513_nsga_quinary_fcc_multi_goal_equalatomic_plot_x_y_"
        data = torch.load( data_path + file_name + str(i) + ".pt")
        data_mapping = []
        for j in range(len(data)):
            data_mapping_x = []
            for k in range(len(data[0])):
                if j == 0:
                    data_mapping_x.append((data[j][k]+9.04)/0.25)
                elif j == 1:
                    data_mapping_x.append((data[j][k]+7.75)/5.88)
            data_mapping.append(data_mapping_x)
        X = data_mapping[0]
        Y = data_mapping[1]
        plt.figure(dpi = 600)
        fig, ax = plt.subplots(figsize=(8,8),dpi = 600)
        ax.set_xlim(80,230)
        ax.set_ylim(1,13)
        X = np.array(X)
        Y = np.array(Y)
        ax.scatter(X, Y, s = 80, marker='o',edgecolors='darkgoldenrod',facecolors='gold', label='NSGA-2:Multi-Objective Optimization')
        #plt.xlabel('F1-Bulk Modulus / GPa')
        plt.xlabel('F1-Bulk Modulus [GPa]')
        #plt.ylabel('F2-Lattice Thermal Conductivity / $\mathrm{W \cdot m^{-1} \cdot K^{-1}}$')
        plt.ylabel('F2-Lattice Thermal Conductivity [$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]')
        #plt.grid()
        plt.title("Generation:{}".format(i))
        plt.show()