# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 19:29:15 2025

@author: Realunknown
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

if __name__ == '__main__' :
    property_name = "lattice_thermal_conduct"
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 32
    
    if property_name == "bulk_modulus":
        data_path = "./result_bulk_modulus_model_0_0.8799_total_data.csv"
        data = pd.read_csv(data_path)
    elif property_name == "lattice_thermal_conduct":
        data_path = "./result_lattice_thermal_conduct_model_0_0.872_total_data.csv"
        data = pd.read_csv(data_path)
        
    #total_data
    """
    data_size = 3
    x_data = data["observed"]
    y_data = data["predicted"]
    data_labels = data["index"]
    labels = ["bi_alloy_fcc","bi_alloy_bcc","bi_alloy_hcp","ter_alloy_fcc","ter_alloy_bcc","ter_alloy_hcp",\
              "qua_alloy_fcc","qua_alloy_bcc","qua_alloy_hcp","qui_alloy_fcc","qui_alloy_bcc","qui_alloy_hcp"]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 9),dpi = 300)
    sns.set_style("white")
    
    # Plot scatter points
    custom_colors = ['#bad3e8', '#f7bc8f', '#b7e4ae'\
                     ,'#93c3d6', '#f6955d', '#8fd68d'\
                      ,'#69a5cb', '#f4743c','#6ac46e'\
                      ,'#457cb6', '#da4722', '#43a354'] #bi_alloy, ter_alloy, qua_alloy,qui_alloy
    palette = custom_colors[:len(labels)]
    for i, label in enumerate(labels):
        mask = data_labels == label
        ax.scatter(x_data[mask], y_data[mask],
                  color=palette[i],
                  label=label,
                  s=300,
                  alpha=0.9,
                  edgecolor='w',
                  linewidth=0.8)
    
    # Axis settings
    if property_name == "bulk_modulus":
        ax.set_title('Total Data', pad=20)
    elif property_name == "lattice_thermal_conduct":
        ax.set_title('Total Data', pad=20)
        ax.set_xlim(0, 30)
    if property_name == "bulk_modulus":
        ax.set_xlabel('Observed [GPa]',  labelpad=12)
        ax.set_ylabel('Predicted [GPa]',  labelpad=12)
    elif property_name == "lattice_thermal_conduct":
        ax.set_xlabel('Observed [$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]',  labelpad=12)
        ax.set_ylabel('Predicted [$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]',  labelpad=12)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Show axis spines (visible by default)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    # Legend and grid
    ax.legend(ncol=4,title='Categories', title_fontsize=16, 
         fontsize=16, framealpha=0.9)
    #ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()
    """
    
    bulk_modulus_list = ["ter_alloy_fcc","ter_alloy_bcc",\
              "qua_alloy_fcc","qua_alloy_bcc","qui_alloy_fcc","qui_alloy_bcc"]
    subspace_name_mapping_bm = {"ter_alloy_fcc":"Ternary FCC","ter_alloy_bcc":"Ternary BCC",\
                             "qua_alloy_fcc":"Quaternary FCC", "qua_alloy_bcc": "Quaternary BCC",\
                            "qui_alloy_fcc":"Quinary FCC", "qui_alloy_bcc": "Quinary BCC"}
    bulk_modulus_color = ['#bad3e8', '#f7bc8f'\
                     ,'#93c3d6', '#f6955d'\
                      ,'#69a5cb', '#f4743c'\
                      ,'#457cb6', '#da4722']
        
    lattice_thermal_conduct_list = ["ter_alloy_fcc",\
              "qua_alloy_fcc","qua_alloy_bcc","qui_alloy_fcc","qui_alloy_bcc"]
    subspace_name_mapping_ltc = {"ter_alloy_fcc":"Ternary FCC","ter_alloy_bcc":"Ternary BCC",\
                             "qua_alloy_fcc":"Quaternary FCC", "qua_alloy_bcc": "Quaternary BCC",\
                            "qui_alloy_fcc":"Quinary FCC", "qui_alloy_bcc": "Quinary BCC"}
    lattice_thermal_conduct_color = ['#bad3e8'\
                     ,'#93c3d6', '#f6955d'\
                      ,'#69a5cb', '#f4743c'\
                      ,'#457cb6', '#da4722']
    
    if property_name == "bulk_modulus":
        i = 0
        for system in bulk_modulus_list:
            fig, ax = plt.subplots(figsize=(12, 9),dpi = 600)
            
            df_sys = data[data['index'] == system]
            x_data = df_sys["observed"]
            y_data = df_sys["predicted"]
            ax.scatter(x_data, y_data,
                      color= bulk_modulus_color[i],
                      s=300,
                      alpha=0.9,
                      edgecolor='w',
                      linewidth=0.8)
            # Plot the scatter points
            ax.set_title(subspace_name_mapping_bm[system], pad=20)
            #ax.set_xlabel('Observed / GPa',  labelpad=12)
            #ax.set_ylabel('Predicted / GPa',  labelpad=12)
            ax.set_xlabel('Observed [GPa]',  labelpad=12)
            ax.set_ylabel('Predicted [GPa]',  labelpad=12)
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            # Show axis spines (visible by default)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            
            # Legend and grid
            #ax.legend(ncol=4,title='Categories', title_fontsize=14, 
            #     fontsize=14, framealpha=0.9)
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            x_reshape = x_data.reshape(-1, 1)  # Convert to a 2D array
            model = LinearRegression().fit(x_reshape, y_data)
            slope = model.coef_[0]
            intercept = model.intercept_
            y_pred = model.predict(x_reshape)
            
            # Calculate the correlation coefficient R²
            r_value = pearsonr(x_data, y_data)[0]
            r_squared = r_value ** 2
            
            # Plot
            if intercept > 0:
                plt.plot(x_data, y_pred, color='#AE6C66', label=f'Fitting results: y={slope:.2f}x+{intercept:.2f}')
            elif intercept < 0:
                intercept = abs(intercept)
                plt.plot(x_data, y_pred, color='#AE6C66', label=f'Fitting results: y={slope:.2f}x-{intercept:.2f}')
            plt.text(0.05, 0.95, f'$r²$ = {r_squared:.4f}', 
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            #ax.grid(True, linestyle='--', alpha=0.4)
            
            plt.legend()
            plt.tight_layout()
            plt.show()
            i += 1
            #break
            
    elif property_name == "lattice_thermal_conduct":
        i = 0
        for system in lattice_thermal_conduct_list:
            
            fig, ax = plt.subplots(figsize=(12, 9),dpi = 600)
            df_sys = data[data['index'] == system]
            x_data = df_sys["observed"]
            y_data = df_sys["predicted"]
            if len(x_data) == 0:
                i += 1
                continue
            ax.scatter(x_data, y_data,
                      color= lattice_thermal_conduct_color[i],
                      s=300,
                      alpha=0.9,
                      edgecolor='w',
                      linewidth=0.8)
            # Plot the scatter points
            ax.set_title(subspace_name_mapping_ltc[system], pad=20)
            ax.set_xlabel('Observed [$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]',  labelpad=12)
            ax.set_ylabel('Predicted [$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]',  labelpad=12)
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            # Show axis spines (visible by default)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            
            # Legend and grid
            #ax.legend(ncol=4,title='Categories', title_fontsize=14, 
            #     fontsize=14, framealpha=0.9)
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            x_reshape = x_data.reshape(-1, 1)  # Convert to a 2D array
            model = LinearRegression().fit(x_reshape, y_data)
            slope = model.coef_[0]
            intercept = model.intercept_
            y_pred = model.predict(x_reshape)
            
            # Calculate the correlation coefficient R²
            r_value = pearsonr(x_data, y_data)[0]
            r_squared = r_value ** 2
            
            # Plot
            
            if intercept > 0:
                plt.plot(x_data, y_pred, color='#AE6C66', label=f'Fitting results: y={slope:.2f}x+{intercept:.2f}')
            elif intercept < 0:
                intercept = abs(intercept)
                plt.plot(x_data, y_pred, color='#AE6C66', label=f'Fitting results: y={slope:.2f}x-{intercept:.2f}')
            plt.text(0.05, 0.95, f'$r²$ = {r_squared:.4f}', 
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            #ax.grid(True, linestyle='--', alpha=0.4)
            
            i += 1
            plt.legend(fontsize=32)
            plt.tight_layout()
            plt.show()
    