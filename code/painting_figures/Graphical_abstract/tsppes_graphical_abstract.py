# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:33:22 2025

@author: Realunknown
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


data_path = "./result_bulk_modulus_model_0_0.8799.csv"
data = pd.read_csv(data_path)

y = []
y_pred = []

for i in range(len(data)):
    if data["system"][i] == "quater_bcc":
        y.append(data["observed"][i])
        y_pred.append(data["ML_predicted"][i])


# Compute prediction interval (simplified version)
"""
residuals = [y[i] - y_pred[i] for i in range(len(y))]
std_err = np.std(residuals)
t_critical = 1.8  # 95% confidence level
margin = t_critical * std_err * np.sqrt(1 + 1/len(y))  # Higher-order terms neglected
"""

# Plot scatter and shaded region
category_colors = {"bi_fcc": '#7890af', "bi_bcc": '#7890af', "bi_hcp": '#7890af',
                   "ter_fcc": '#cb9475',"ter_bcc":'#cb9475',"ter_hcp": '#cb9475',
                   "quater_fcc":'#8cbf87',"quater_bcc":'#8cbf87',"quater_hcp":'#8cbf87',
                   "qui_fcc":'#af6d66',"qui_bcc":'#af6d66',"qui_hcp":'#af6d66',
                   }
#point_colors = [category_colors[ele] for ele in data["system"]]
point_colors = '#808080'

fig, ax = plt.subplots(dpi=600,figsize = (5,5))
ax.scatter(y, y_pred,color = point_colors)
#plt.plot([min(y), max(y)], [min(y_pred), max(y_pred)],color = "#69413d", label="Ideal Line")

ax.set_xticks([])
ax.set_yticks([])

"""
plt.scatter(y, y_pred, alpha=0.5, label="Predicted vs True", color=point_colors)
plt.plot([min(y), max(y)], [min(y_pred), max(y_pred)], 'r--', label="Ideal Line")
plt.fill_between(
    sorted(y), 
    y_pred[np.argsort(y)] - margin, 
    y_pred[np.argsort(y)] + margin, 
    color='gray', 
    alpha=0.3,
    label="95% Prediction Interval"
)
"""

#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
#ax.spines['left'].set_position(('data', 0))
#ax.spines['bottom'].set_position(('data', 0))
#ax.annotate('', xy=(1, 0), xytext=(0, 0),arrowprops=dict(arrowstyle="->"))
#ax.annotate('', xy=(0, 1), xytext=(0, 0),arrowprops=dict(arrowstyle="->"))
plt.ylim(-10,50)
#plt.xlabel("True Values")
#plt.ylabel("Predicted Values")
#plt.legend()
plt.savefig("./ter_bcc.png",dpi =600)
plt.show()