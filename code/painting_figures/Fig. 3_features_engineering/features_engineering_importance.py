# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:46:42 2026

@author: Realunknown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

target = "lattice_thermal_conduct"
#target = "bulk_modulus"
# ---------------------- Read your data ----------------------
df = pd.read_csv("features_thermal_conductivity_with_label.csv")
#df = pd.read_csv("features_bulk_modulus_with_label.csv")

# ---------------------- Data cleaning ----------------------
X = df.drop(["material_id"], axis=1)
X = X.select_dtypes(include=[np.number])
y = X["target"]
X = X.drop(["target"], axis=1)

# ---------------------- Compute Pearson correlation coefficients ----------------------
corr_with_target = X.corrwith(y)
corr_df = pd.DataFrame(corr_with_target, columns=["Pearson Correlation"])
# Sort by correlation coefficient
corr_df = corr_df.sort_values(by="Pearson Correlation", ascending=False)

# ---------------------- Horizontal bar chart layout ----------------------
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 22
plt.figure(figsize=(14, 10),dpi = 600)  # horizontal widescreen layout

# draw the horizontal bar chart
if target == "lattice_thermal_conduct":
    bars = sns.barplot(
        x=corr_df["Pearson Correlation"],
        y=corr_df.index,
        palette="vlag",
        edgecolor="white"
    )
elif target == "bulk_modulus":
    bars = sns.barplot(
        x=corr_df["Pearson Correlation"],
        y=corr_df.index,
        palette="coolwarm",
        edgecolor="white"
    )

# annotate values on the bars
for bar in bars.patches:
    width = bar.get_width()
    label_x_pos = width + 0.02 if width > 0 else width - 0.01
    bars.text(
        label_x_pos,
        bar.get_y() + bar.get_height()/2,
        f"{width:.2f}",
        va="center",
        fontsize=20
    )

# styling
plt.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
plt.xlim(-0.65,0.75)
plt.xlabel("Pearson Correlation Coefficient", fontsize=28)
plt.ylabel("Features", fontsize=28)
if target == "lattice_thermal_conduct":
    plt.title("Lattice Thermal Conductivity Feature Importance", fontsize=28, pad=15)
    #plt.savefig("pearson_correlation_horizontal_ltc.png", dpi=600, bbox_inches="tight")
elif target == "bulk_modulus":
    plt.title("Bulk Modulus Feature Importance", fontsize=28, pad=15)
    #plt.savefig("pearson_correlation_horizontal_bm.png", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()