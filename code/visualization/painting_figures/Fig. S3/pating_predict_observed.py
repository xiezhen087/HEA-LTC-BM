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

property_name = "lattice_thermal_conduct"

if property_name == "bulk_modulus":
    data_path = "./data/bulk_modulus/"  # TODO: set your local data path
    x_observed = torch.load(data_path + "x_observed-0-0.154.pt")
    y_predicted = torch.load(data_path + "y_predicted-0-0.154.pt")
elif property_name == "lattice_thermal_conduct":
    data_path = "./data/lattice_thermal_conduct/"  # TODO: set your local data path
    x_observed = torch.load(data_path + "x_observed-0-0.219.pt")
    y_predicted = torch.load(data_path + "y_predicted-0-0.219.pt")
    
data = pd.DataFrame({'x_observed':x_observed, 'y_predicted': y_predicted})
r_square = r2_score(x_observed,y_predicted)
# Set up the canvas
plt.figure(figsize=(6, 6),dpi = 600)

# Scatter plot of observed versus predicted values
if property_name == "bulk_modulus":
    sns.scatterplot(data = data, x = "x_observed",y="y_predicted", color='#3E608D', label = "Validation set R² : {:.4f}".format(r_square))
elif property_name == "lattice_thermal_conduct":
    sns.scatterplot(data = data, x = "x_observed",y="y_predicted", color='#CB9475', label = "Validation set R² : {:.4f}".format(r_square))
plt.plot(x_observed, x_observed, color='#8D2F25', linestyle='--')  # perfect prediction line

# Add labels
if property_name == "bulk_modulus":
    plt.title('Bulk Modulus in pure metals and alloys')
    #plt.xlabel('Observed Values / log(GPa)')
    #plt.ylabel('Predicted Values / log(GPa)')
    plt.xlabel('Observed Values [log(GPa)]')
    plt.ylabel('Predicted Values [log(GPa)]')
elif property_name == "lattice_thermal_conduct":
    plt.title('Lattice Thermal Conductivity in pure metals and alloys')
    #plt.xlabel('Observed Values / log($\mathrm{W \cdot m^{-1} \cdot K^{-1}}$)')
    #plt.ylabel('Predicted Values / log($\mathrm{W \cdot m^{-1} \cdot K^{-1}}$)')
    plt.xlabel('Observed Values [log($\mathrm{W \cdot m^{-1} \cdot K^{-1}}$)]')
    plt.ylabel('Predicted Values [log($\mathrm{W \cdot m^{-1} \cdot K^{-1}}$)]')
if property_name == "bulk_modulus":
    plt.savefig("./Arial/bulk_modulus_in_pure_metals_and_alloys.png",dpi =600)
elif property_name == "lattice_thermal_conduct":
    plt.savefig("./Arial/lattice_thermal_conduct_in_pure_metals_and_alloys.png",dpi =600)
plt.legend()
#plt.grid()

# Display the figure
plt.show()