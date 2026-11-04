# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:31:37 2026

@author: Realunknown
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

# ---------------------- 1. Generate pseudo-data (faithful reproduction: one quinary alloy corresponds to 5 element points) ----------------------
# 1.1 Define X-axis elements (grouped by periodic table groups)
elements = [
    # 3d transition metal group
    'Fe', 'Co', 'Ni', 'Mn', 'Cu',
    # 4d transition metal group
    'Pd', 'Ag', 'Cd', 'Rh', 'Ru',
    # Main group metal group
    'Zn', 'Mg', 'Al', 'Sn', 'Pb'
]
element_set = set(elements)
n_elements = len(elements)

# 1.2 Define quinary alloy systems (each system consists of 5 different elements)
alloy_systems = [
    'Zn-Ni-Mg-Mn-Co',  # corresponding elements: Zn,Ni,Mg,Mn,Co
    'Fe-Co-Ni-Mn-Cu',  # corresponding elements: Fe,Co,Ni,Mn,Cu
    'Pd-Ag-Cd-Rh-Ru',  # corresponding elements: Pd,Ag,Cd,Rh,Ru
    'Zn-Mg-Al-Sn-Pb',  # corresponding elements: Zn,Mg,Al,Sn,Pb
    'Fe-Ni-Mn-Cu-Zn',  # corresponding elements: Fe,Ni,Mn,Cu,Zn
    'Co-Ni-Mn-Cu-Mg',  # corresponding elements: Co,Ni,Mn,Cu,Mg
    'Fe-Co-Mn-Cu-Al',  # corresponding elements: Fe,Co,Mn,Cu,Al
    'Ni-Mn-Cu-Zn-Mg',  # corresponding elements: Ni,Mn,Cu,Zn,Mg
    'Co-Mn-Cu-Zn-Al',  # corresponding elements: Co,Mn,Cu,Zn,Al
    'Mn-Cu-Zn-Mg-Al',  # corresponding elements: Mn,Cu,Zn,Mg,Al
]
n_alloys = len(alloy_systems)

# 1.3 Generate pseudo-data: one quinary alloy -> one point at each of the 5 element positions (same color, same thermal conductivity)
data = []
for alloy in alloy_systems:
    # Split the 5 elements of the alloy system
    alloy_elements = alloy.split('-')
    # Generate a unique thermal conductivity value for this alloy (fixed value, ensuring the 5 points have the same value)
    kappa = np.random.normal(loc=4, scale=1.5)
    kappa = np.clip(kappa, 0, 8)  # limit the range to 0~8
    
    # Generate one data point for this alloy at each of the 5 element positions
    for elem in alloy_elements:
        if elem in element_set:  # ensure the element is in the X-axis list
            data.append({
                'Element': elem,
                'Alloy_System': alloy,
                'Lattice_Thermal_Conductivity': kappa
            })

# Convert to DataFrame
df = pd.DataFrame(data)

# ---------------------- 2. Visualization configuration ----------------------
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'Arial',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'figure.figsize': (16, 10)  # leave space at the bottom for the legend
})

# 2.1 Define color scheme: each alloy system corresponds to a unique color (tab20 high contrast)
cmap = ListedColormap(plt.cm.tab20.colors[:n_alloys])  # pick colors according to the number of alloys
alloy2color = {alloy: cmap(i) for i, alloy in enumerate(alloy_systems)}

# 2.2 Create the canvas (reserve bottom space for the legend)
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# ---------------------- 3. Plot scatter (core: one alloy -> 5 same-colored points) ----------------------
# Iterate over alloy systems (ensure the 5 points of the same alloy have exactly the same color)
for alloy in alloy_systems:
    alloy_data = df[df['Alloy_System'] == alloy]
    # Get the 5 element positions and thermal conductivity of this alloy
    x = [elements.index(elem) for elem in alloy_data['Element']]
    y = alloy_data['Lattice_Thermal_Conductivity'].values
    # Use a fixed color for the same alloy
    color = alloy2color[alloy]
    
    # Small points + semi-transparent to avoid overlap; add a light edge to improve visibility
    ax.scatter(x, y, s=20, c=color, alpha=0.8, edgecolors='white', linewidth=0.2)

# ---------------------- 4. Axis optimization ----------------------
# 4.1 X-axis: grouped layout + rotated labels + group annotations
ax.set_xticks(np.arange(n_elements))
ax.set_xticklabels(elements, rotation=45, ha='right', fontsize=9)
# Group text annotations
ax.text(2, -0.8, '3d Transition Metals', ha='center', fontsize=10, fontweight='bold')
ax.text(7, -0.8, '4d Transition Metals', ha='center', fontsize=10, fontweight='bold')
ax.text(12, -0.8, 'Main Group Metals', ha='center', fontsize=10, fontweight='bold')
# Group separator lines
ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=9.5, color='gray', linestyle='--', alpha=0.5)

# 4.2 Y-axis: clipped range + dense ticks + grid
ax.set_ylim(0, 8)
ax.set_yticks(np.arange(0, 8.1, 1))
ax.tick_params(axis='y', labelsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

# ---------------------- 5. Legend optimization (below + spanning full width) ----------------------
n_legend_cols = 5  # 5-column layout, fits 10 alloy systems, spans the full chart width
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=alloy2color[alloy],
               markersize=8, alpha=0.8, label=alloy, markeredgecolor='white', markeredgewidth=0.2)
    for alloy in alloy_systems
]
ax.legend(
    handles=legend_handles,
    ncol=n_legend_cols,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.2),
    fontsize=8,
    frameon=False,
    title='Quinary Alloy Systems',
    title_fontsize=10,
    columnspacing=1.5,
    handletextpad=0.8
)

# ---------------------- 6. Labels and title ----------------------
ax.set_xlabel('Constituent Elements', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Lattice Thermal Conductivity (W·m⁻¹·K⁻¹)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Distribution of Lattice Thermal Conductivity in Quinary FCC HEAs', 
             fontsize=14, fontweight='bold', pad=20)

# Save high-resolution figure
plt.savefig('HEA_Thermal_Conductivity_Map_Correct.png', dpi=300, bbox_inches='tight')
plt.show()
