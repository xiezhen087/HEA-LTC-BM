# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 18:56:20 2025

@author: Realunknown
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

#plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16

property_name = "bulk_modulus"

if property_name == "bulk_modulus":
    data_path_cgcnn = "./data/cgcnn_solo/"  # TODO: set your local data path
    data_path_mix = "./data/mix_cgcnn/"  # TODO: set your local data path
    x_observed_cgcnn = torch.load(data_path_cgcnn + "x_observed_K_VRH_23040401.pt")
    y_predicted_cgcnn = torch.load(data_path_cgcnn + "y_predicted_K_VRH_23040401.pt")
    x_observed_mix = torch.load(data_path_mix + "x_observed-0-0.879.pt")
    y_predicted_mix = torch.load(data_path_mix + "y_predicted-0-0.879.pt")
elif property_name == "lattice_thermal_conduct":
    data_path_cgcnn = "./data/cgcnn_solo/"  # TODO: set your local data path
    data_path_mix = "./data/mix_cgcnn/"  # TODO: set your local data path
    x_observed_cgcnn = torch.load(data_path_cgcnn + "x_observed_23040616.pt")
    y_predicted_cgcnn = torch.load(data_path_cgcnn + "y_predicted_23040616.pt")
    x_observed_mix = torch.load(data_path_mix + "x_observed-0-0.871.pt")
    y_predicted_mix = torch.load(data_path_mix + "y_predicted-0-0.871.pt")
    
data_cgcnn = pd.DataFrame({'x_observed':x_observed_cgcnn, 'y_predicted': y_predicted_cgcnn})
data_mix = pd.DataFrame({'x_observed':x_observed_mix, 'y_predicted': y_predicted_mix})
r_square_cgcnn = r2_score(x_observed_cgcnn,y_predicted_cgcnn)
r_square_mix = r2_score(x_observed_mix,y_predicted_mix)
# Set up the canvas
plt.figure(figsize=(6, 6),dpi = 600)

# Scatter plot of observed versus predicted values
if property_name == "bulk_modulus":
    sns.scatterplot(data = data_cgcnn, x = "x_observed",y="y_predicted", color='#9BBBE1', label = "CGCNN Validation set R² : {:.4f}".format(r_square_cgcnn))
    sns.scatterplot(data = data_mix, x = "x_observed",y="y_predicted", color='#3E608D', label = "Mix-CGCNN Validation set R² : {:.4f}".format(r_square_mix))
elif property_name == "lattice_thermal_conduct":
    sns.scatterplot(data = data_cgcnn, x = "x_observed",y="y_predicted", color='#eec69c', label = "CGCNN Validation set R² : {:.4f}".format(r_square_cgcnn))
    sns.scatterplot(data = data_mix, x = "x_observed",y="y_predicted", color='#CB9475', label = "Mix-CGCNN Validation set R² : {:.4f}".format(r_square_mix))
plt.plot(x_observed_mix, x_observed_mix, color='#8D2F25', linestyle='--')  # perfect prediction line

# Add labels
plt.legend(loc='upper left')
if property_name == "bulk_modulus":
    plt.title('Bulk Modulus by CGCNN and Mix-CGCNN')
    #plt.xlabel('Observed Values / log(GPa)')
    #plt.ylabel('Predicted Values / log(GPa)')
    plt.xlabel('Observed Values [log(GPa)]')
    plt.ylabel('Predicted Values [log(GPa)]')
elif property_name == "lattice_thermal_conduct":
    plt.title('Lattice Thermal Conductivity by CGCNN and Mix-CGCNN')
    plt.xlabel('Observed Values [log($\mathrm{W \cdot m^{-1} \cdot K^{-1}}$)]')
    plt.ylabel('Predicted Values [log($\mathrm{W \cdot m^{-1} \cdot K^{-1}}$)]')
if property_name == "bulk_modulus":
    plt.savefig("./Arial/bulk_modulus_by_CGCNN_and_Mix_CGCNN.png",dpi =600)
elif property_name == "lattice_thermal_conduct":
    plt.savefig("./Arial/lattice_thermal_conduct_by_CGCNN_and_Mix_CGCNN.png",dpi =600)
#plt.grid()

# Display the figure
plt.show()