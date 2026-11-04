# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:59:42 2025

@author: Realunknown
"""
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.pyplot as plt


import seaborn as sns

def chi2_analysis(df, data_type, target_col='label'):
    # Ensure feature values are non-negative (required by the chi-squared test)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.drop(target_col, axis=1))
    y = df[target_col].values
    
    # Compute chi-squared statistics and p-values
    chi_scores, p_values = chi2(X, y)
    
    # Create the result DataFrame
    features = df.columns.drop(target_col)
    result_df = pd.DataFrame({
        'Feature': features,
        'Chi2_Score': chi_scores,
        'P_Value': p_values
    }).sort_values('Chi2_Score', ascending=False)
    
    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.bar(result_df['Feature'], result_df['Chi2_Score'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Chi-Squared Score')
    plt.title('Feature Importance by Chi-Squared Test in ' + data_type)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__' :
    data_path = "./data/Fig. 3_features_engineering/"  # TODO: set your local data path
    
    head = ["material_id","density","space_group","lattice_type","volume_pre_atom","volume_pre_cell","variance_of_atom_radius","extreme_atom_radius_mean_z_score","number_of_species","number_of_atom_per_cell","is_gas","is_metal","metal_elements_proportion","magnetic_elements_proportion","fw_atomic_number","fw_valence_electrons","fw_atomic_mass","fw_group","fw_period","fw_electronegativity","fw_boiling_point","fw_melting_point","fw_van der_Waals radii","fw_atomic_radius_calculated","fw_electrical_resistivity","fw_velocity_of_sound","fw_thermal_conductivity","fw_bulk_modulus","fw_youngs_modulus","fw_coefficient_of_linear_thermal_expansion","fw_ionization_energies","target"]
    
    property_name = "thermal_conductivity"
    
    csv_path_time = data_path + "features_" + property_name + "_with_label.csv"
    df_time = pd.read_csv(csv_path_time)
    df_time = df_time.drop(columns=df_time.columns[0])
    
    """
    #violinplot
    
    plt.rcParams.update({'font.size': 18})
    f, axes = plt.subplots(3, 5, figsize=(20, 12), sharex=True)
    sns.despine(left=True)
    
    antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864'] 
    
    for i in range(3):
        for j in range(5):
            if i == 2 and j == 4:
                axes[i, j].axis('off')
                continue
            sns.violinplot(x='label', y=time_head[i*5+j] , data=df_time, palette=antV, ax=axes[(i, j)])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8)
    plt.show()
    """
    
    #colinearity
    df_pure_time = df_time.drop(columns=df_time.columns[-1])
    plt.figure(figsize=(25, 25),dpi = 300)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)

    corrMatt = df_pure_time.corr()
    mask = np.array(corrMatt)
    mask[np.triu_indices_from(mask)] = False #lower triangle
    h = sns.heatmap(corrMatt, mask=mask,vmax=1,cbar=False, square=True,annot=True,annot_kws={"size": 28},fmt='.2f',cmap="viridis",cbar_kws={'ticks': [-1, 0, 1]})
    cb = h.figure.colorbar(h.collections[0]) #show colorbar
    cb.ax.tick_params(labelsize=40)  # set colorbar tick font size
    plt.xticks(rotation=75)
    plt.show()
    
    """
    #Chi square test
    plt.rcParams.update({'font.size': 18})
    data_type = "Time Domain"
    chi2_analysis(df_time, data_type , target_col='label')
    """
    
    """
    #freq
    #violinplot
    freq_head = ["df_amp_mean","df_amp_var","df_amp_skew","df_amp_kurt","df_freq_mean"," df_freq_rms"," df_freq_var"," df_freq_main"," df_energy_concentration"," df_freq_skew","df_freq_kurt","df_mean_normalized_spectrum","label"]
    
    csv_path_freq = data_path + "features_freq_domain_best_right_partion.csv"
    df_freq = pd.read_csv(csv_path_freq)
    df_freq = df_freq.drop(columns=df_freq.columns[0])
    
    #violinplot
    plt.rcParams.update({'font.size': 18})
    f, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
    sns.despine(left=True)
    
    antV = [ '#8543E0', '#13C2C2', '#3436c7', '#F04864'] 
    
    for i in range(3):
        for j in range(4):
            sns.violinplot(x='label', y=freq_head[i*4+j] , data=df_freq, palette=antV, ax=axes[(i, j)])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8)
    plt.show()
    
    
    #colinearity 
    plt.figure(figsize=(25, 25),dpi = 300)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    
    df_pure_freq = df_freq.drop(columns=df_freq.columns[-1])
    corrMatt = df_pure_freq.corr()
    mask = np.array(corrMatt)
    mask[np.triu_indices_from(mask)] = False #lower triangle
    h = sns.heatmap(corrMatt, mask=mask,vmax=1,cbar=False, square=True,annot=True,annot_kws={"size": 28},fmt='.2f',cmap="viridis",cbar_kws={'ticks': [-1, 0, 1]})
    cb = h.figure.colorbar(h.collections[0]) #show colorbar
    cb.ax.tick_params(labelsize=40)  # set colorbar tick font size
    plt.xticks(rotation=75)
    plt.show()
    
    
    #Chi square test
    plt.rcParams.update({'font.size': 18})
    data_type = "Frequency Domain"
    chi2_analysis(df_freq, data_type , target_col='label')
    """