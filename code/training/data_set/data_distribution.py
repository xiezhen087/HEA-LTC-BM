# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:17:08 2024

@author: Realunknown
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_path = "./thermal_conduct/"

if __name__ == '__main__' :
    #data_mlp = torch.load(data_path + "data_for_mlp.pt")
    #df_ini = pd.DataFrame(data_mlp, columns = header)
    #df = pd.DataFrame(data_mlp, columns = header)
    df = pd.read_csv(data_path + 'features_thermal_conductivity_with_label.csv')
    df["extreme_atom_radius_mean_z_score_log"] = df["extreme_atom_radius_mean_z_score"].apply(np.log10)
    sns.boxplot(x='extreme_atom_radius_mean_z_score', data=df)
    plt.show()