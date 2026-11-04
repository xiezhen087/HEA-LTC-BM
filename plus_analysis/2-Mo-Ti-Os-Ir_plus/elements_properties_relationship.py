# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:36:51 2025

@author: Realunknown

kimi (2026-08-05): added marginal-distribution panels for selected elements
(FCC: Os, Ir; BCC: Mo, Ti) on top of the original pie-sector scatter.
Original de-duplication (remove_duplicate), pie-sector drawing and element
coloring are kept unchanged. Marginal curves are element-fraction-weighted
KDEs of the mapped B / kappa values along the two axes.
Also fixed the variable names in value_mapping (bm/ltc -> value_bm/value_ltc)
and pointed structure_path to the local ./FCC and ./BCC folders.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pymatgen.io.vasp import Poscar
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

#Old color
"""
element_color = {"Ti":"#d7dee7", "V":"#c9d3df", "Cr": "#bcc8d7" ,"Mn": "#aebccf", "Fe" : "#a1b1c7", "Co" : "#93a6bf", "Ni": "#869bb7", "Cu": "#7890af", "Zn": "#6c829e",
    "W": "#efdfd6", "Ta": "#ead4c8", "Mo": "#e5caba", "Nb": "#e0bfac", "Zr" : "#dbb49e", "Hf" : "#d5a991", "Re":"#d09f83",
    "Al": "#ddecdb", "Mg": "#d1e5cf", "Be" : "#c6dfc3", "K": "#bad9b7", "Na" : "#afd2ab", "Sr" : "#a3cc9f", "Ba": "#98c593", "Ca": "#8cbf87", "Li":"#7eac7a",
    "Sn": "#c79994", "In":"#bf8a85", "Bi": "#b77c75", "Pb":"#af6d66",
    "Au": "#eae3e9", "Ag": "#e3dae1", "Pt" : "#dcd1da", "Pd":"#d4c8d3", "Ir": "#cdbfcb", "Os": "#c6b5c4", "Rh": "#bfacbc", "Ru":"#b8a3b5",
    "Sc": "#e8e8e8", "Y": "#e0e1e0", "La": "#d8d9d9", "Ce": "#d0d1d1", "Pr": "#c8cac9", "Nd": "#c1c2c1", "Pm":"#b9bbba", "Sm": "#b1b3b2", "Eu": "#9fa1a0", "Gd" : "#8e8f8e", "Tb": "#7c7d7d",
    "Dy": "#6a6b6b", "Ho": "#595a59", "Er" : "#474847", "Tm":"#353635", "Yb": "#f7f7f7", "Lu": "#eff0f0",
    "Tc": "#000000"
    }
"""

#New color
element_color = {"Ti":"#bcd6e9", "V":"#8fbbda", "Cr": "#81a8c4" ,"Mn": "#79add2", "Fe" : "#62a0cb", "Co": "#4c92c3", "Ni": "#3585bc", "Cu": "#1f77b4", "Zn": "#1c6ba2",
    "W": "#ffb26e", "Ta": "#cc660b", "Mo": "#FCD364", "Nb": "#ff8c26", "Zr" : "#EF5D1D", "Hf" : "#C93006", "Re":"#7a3d07",
    "Al": "#d5ecd5", "Mg": "#c0e3c0", "Be" : "#abd9ab", "K": "#96d096", "Na" : "#80c680", "Sr" : "#6bbd6b", "Ba": "#56b356", "Ca": "#41aa41", "Li":"#2ca02c",
    "Sn": "#c79994", "In":"#bf8a85", "Bi": "#b77c75", "Pb":"#af6d66",
    "Au": "#dfd1eb", "Ag": "#d4c2e5", "Pt" : "#b291d0", "Pd":"#8f6ab0", "Ir": "#e2d6ed", "Os": "#c5addc", "Rh": "#6f5389", "Ru":"#503b62",
    "Sc": "#ddccc9", "Y": "#d1bbb7", "La": "#c6aba5", "Ce": "#ba9a93", "Pr": "#af8981", "Nd": "#a3786f", "Pm":"#98675d", "Sm": "#8c564b", "Eu": "#7e4d44", "Gd" : "#70453c", "Tb": "#623c35",
    "Dy": "#54342d", "Ho": "#462b26", "Er" : "#38221e","Tm" : "#2a1a17", "Yb": "#1c110f", "Lu": "#e8dddb",
    "Tc": "#000000"
    }

#element_color_fcc = {"W": "#636e72", "Pd" :"#b2bec3", "Cr": "#fdcb6e", "Ti": "#fd79a8", "Rh": "#0984e3", "Os": "#00cec9", "Re":"#a29bfe", "Ta": "#d63031", "Ru": "#00b894", "Pt":"#e17055", "Ir":"#74b9ff"}

# ---- kimi: target elements for the marginal distributions and their line colors
TARGET_ELEMENTS = {"FCC": ["Os", "Ir"], "BCC": ["Mo", "Ti"]}
MARGINAL_COLOR = {"Os": "#503b62", "Ir": "#b291d0", "Mo": "#cc660b", "Ti": "#3585bc"}

def remove_duplicate(structure_file_list):
    structure_file_list_new = []
    element_label = []
    for structure_file in structure_file_list:
        stru_list = structure_file.split("_")
        ele = stru_list[0] + "-" + stru_list[1]
        if ele not in element_label:
            element_label.append(ele)
            structure_file_list_new.append(structure_file)
    return structure_file_list_new

def value_mapping(value_bm, value_ltc, subspace):
    if subspace == "FCC":
        bm_mapped = (value_bm + 9.04)/0.25
        ltc_mapped = (value_ltc + 7.75)/5.88
    elif subspace == "BCC":
        bm_mapped = (value_bm + 2.38)/0.13
        ltc_mapped = (value_ltc + 5.03)/1.28
    return bm_mapped, ltc_mapped

# ---- kimi: element-fraction-weighted KDE helper
def weighted_kde(values, weights, grid):
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    if weights.sum() <= 0 or len(np.unique(values)) < 2:
        return np.zeros_like(grid)
    kde = gaussian_kde(values, weights=weights)
    return kde(grid)

if __name__ == '__main__' :
    script_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = script_dir  # this script sits at the task folder root
    out_fig_dir = os.path.join(task_dir, "final_fig")
    out_data_dir = os.path.join(task_dir, "final_data")
    os.makedirs(out_fig_dir, exist_ok=True)
    os.makedirs(out_data_dir, exist_ok=True)

    for subspace in ["FCC", "BCC"]:
        structure_path = os.path.join(task_dir, subspace) + "/"
        structure_file_list = os.listdir(structure_path)
        structure_file_list = remove_duplicate(structure_file_list)
        plt.rcParams.update({
        #'font.family': 'Times New Roman',
        'font.family': 'Arial',
        'font.size': 24,
    })

        element_list = []
        ratio_list = []
        ratio_add_up_list = []
        bm_function_list = []
        ltc_function_list = []
        target_frac = {ele: [] for ele in TARGET_ELEMENTS[subspace]}   # kimi
        for structure_file in structure_file_list:
            poscar = Poscar.from_file(structure_path + structure_file)
            structure = poscar.structure

            #elements_ratio
            elements = [atom.specie.symbol for atom in structure]
            element_type = list(set(elements))
            elements_ratio = [elements.count(ele)/len(elements) for ele in element_type]
            ratio_list.append(elements_ratio)
            ratio_add_up = [elements_ratio[0], sum(elements_ratio[0:2]), sum(elements_ratio[0:3]), sum(elements_ratio[0:4])]

            pro = structure_file.split("_")
            bm = float(pro[-2])
            ltc = float(pro[-1])
            bm_mapped, ltc_mapped = value_mapping(bm, ltc, subspace)
            ltc_function_list.append(ltc_mapped)
            bm_function_list.append(bm_mapped)

            element_list.append(element_type)
            ratio_add_up_list.append(ratio_add_up)

            # ---- kimi: atomic fraction of each target element in this alloy
            for ele in TARGET_ELEMENTS[subspace]:
                if ele in element_type:
                    target_frac[ele].append(elements_ratio[element_type.index(ele)])
                else:
                    target_frac[ele].append(0.0)

        # ===================== kimi: GridSpec layout (main pie scatter + marginals) =====================
        fig = plt.figure(figsize=(11, 10.6), dpi=600)
        gs = GridSpec(4, 4, figure=fig, hspace=0.06, wspace=0.06,
                      left=0.16, right=0.88, top=0.84, bottom=0.12)
        ax = fig.add_subplot(gs[1:, :-1])
        ax_top = fig.add_subplot(gs[0, :-1], sharex=ax)
        ax_right = fig.add_subplot(gs[1:, -1], sharey=ax)

        i=0
        while i<len(bm_function_list):
            x = [0] + np.cos(np.linspace(0, 2 * np.pi * ratio_add_up_list[i][0], 30)).tolist()
            y = [0] + np.sin(np.linspace(0, 2 * np.pi * ratio_add_up_list[i][0], 30)).tolist()
            xy1 = list(zip(x, y))

            x = [0] + np.cos(np.linspace(2 * np.pi * ratio_add_up_list[i][0], 2 * np.pi * ratio_add_up_list[i][1], 30)).tolist()
            y = [0] + np.sin(np.linspace(2 * np.pi * ratio_add_up_list[i][0], 2 * np.pi * ratio_add_up_list[i][1], 30)).tolist()
            xy2 = list(zip(x, y))

            x = [0] + np.cos(np.linspace(2 * np.pi * ratio_add_up_list[i][1], 2 * np.pi* ratio_add_up_list[i][2], 30)).tolist()
            y = [0] + np.sin(np.linspace(2 * np.pi * ratio_add_up_list[i][1], 2 * np.pi* ratio_add_up_list[i][2], 30)).tolist()

            xy3 = list(zip(x, y))

            x = [0] + np.cos(np.linspace(2 * np.pi * ratio_add_up_list[i][2], 2 * np.pi*1, 30)).tolist()
            y = [0] + np.sin(np.linspace(2 * np.pi * ratio_add_up_list[i][2], 2 * np.pi*1, 30)).tolist()

            xy4 = list(zip(x, y))

            x = [0] + np.cos(np.linspace(2 * np.pi * ratio_add_up_list[i][3], 2 * np.pi*1, 30)).tolist()
            y = [0] + np.sin(np.linspace(2 * np.pi * ratio_add_up_list[i][3], 2 * np.pi*1, 30)).tolist()

            xy5 = list(zip(x, y))

            xy=[xy1,xy2,xy3,xy4,xy5]
            c=[element_color[element_list[i][0]],element_color[element_list[i][1]],element_color[element_list[i][2]],element_color[element_list[i][3]],element_color[element_list[i][4]]]
            for j in range(5):
                ax.scatter(bm_function_list[i], ltc_function_list[i], marker=(xy[j]), s=2000,facecolor=c[j])
            i=i+1
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)

        # ===================== kimi: marginal distribution panels =====================
        bm_arr = np.array(bm_function_list)
        ltc_arr = np.array(ltc_function_list)
        x_grid = np.linspace(bm_arr.min(), bm_arr.max(), 300)
        y_grid = np.linspace(ltc_arr.min(), ltc_arr.max(), 300)
        if len(np.unique(bm_arr)) > 1:
            ax_top.plot(x_grid, gaussian_kde(bm_arr)(x_grid), "--", color="#7f7f7f", lw=1.8, label="All alloys")
        if len(np.unique(ltc_arr)) > 1:
            ax_right.plot(gaussian_kde(ltc_arr)(y_grid), y_grid, "--", color="#7f7f7f", lw=1.8, label="All alloys")
        for ele in TARGET_ELEMENTS[subspace]:
            w = target_frac[ele]
            c_m = MARGINAL_COLOR[ele]
            dx = weighted_kde(bm_arr, w, x_grid)
            dy = weighted_kde(ltc_arr, w, y_grid)
            ax_top.plot(x_grid, dx, color=c_m, lw=2.6, label=f"{ele}-weighted")
            ax_top.fill_between(x_grid, 0, dx, color=c_m, alpha=0.12)
            ax_right.plot(dy, y_grid, color=c_m, lw=2.6, label=f"{ele}-weighted")
            ax_right.fill_betweenx(y_grid, 0, dy, color=c_m, alpha=0.12)
        # kimi: single legend row above the top marginal panel (avoids overlapping curves)
        handles, labels_leg = ax_top.get_legend_handles_labels()
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], marker='o', ls='', markersize=12,
                              markerfacecolor='none', markeredgecolor='black',
                              markeredgewidth=2.2))
        labels_leg.append("This work (experimental)")
        fig.legend(handles, labels_leg, loc="upper center",
                   bbox_to_anchor=(0.52, 0.895), ncol=4, fontsize=15,
                   frameon=False, columnspacing=1.6, handlelength=2.2)
        ax_top.set_ylabel("Density", fontsize=16)
        ax_right.set_xlabel("Density", fontsize=16)
        ax_top.tick_params(axis='y', labelsize=12)
        ax_right.tick_params(axis='x', labelsize=12)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)

        # ===================== kimi: overlay experimental alloys from ./EXP =====================
        # same pie-sector drawing as the optimization alloys, plus a black ring
        # and a bold name tag so readers can spot the experimental materials
        exp_dir = os.path.join(task_dir, "EXP")
        exp_tags = {"FCC": ["Pt-1"], "BCC": ["RE-3", "RE-4"]}[subspace]
        exp_bm, exp_ltc = [], []
        for tag in exp_tags:
            for fname in os.listdir(exp_dir):
                if fname.split("_")[1] != tag:
                    continue
                pro = fname.split("_")
                bm_e, ltc_e = value_mapping(float(pro[-2]), float(pro[-1]), subspace)
                exp_bm.append(bm_e)
                exp_ltc.append(ltc_e)
                structure = Poscar.from_file(os.path.join(exp_dir, fname)).structure
                elements_e = [atom.specie.symbol for atom in structure]
                element_type_e = list(set(elements_e))
                ratio_e = [elements_e.count(ele)/len(elements_e) for ele in element_type_e]
                rau_e = [sum(ratio_e[:k+1]) for k in range(4)]

                xy_e = []
                bounds = [0.0] + rau_e + [1.0]
                for j in range(5):
                    x = [0] + np.cos(np.linspace(2 * np.pi * bounds[j], 2 * np.pi * bounds[j+1], 30)).tolist()
                    y = [0] + np.sin(np.linspace(2 * np.pi * bounds[j], 2 * np.pi * bounds[j+1], 30)).tolist()
                    xy_e.append(list(zip(x, y)))
                c_e = [element_color[element_type_e[j]] for j in range(5)]
                for j in range(5):
                    ax.scatter(bm_e, ltc_e, marker=(xy_e[j]), s=2000, facecolor=c_e[j], zorder=4)
                # black ring + bold tag
                ax.scatter(bm_e, ltc_e, s=3800, facecolor='none', edgecolor='black',
                           linewidth=2.4, zorder=5)
                ax.annotate(tag, (bm_e, ltc_e), textcoords="offset points",
                            xytext=(0, 26), ha='center', fontsize=16,
                            fontweight='bold', zorder=6,
                            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                      ec='none', alpha=0.75))

        # kimi: include experimental points and their ring/label padding in the
        # displayed range, rather than relying on autoscaling of the alloy cloud
        x_all = np.asarray(bm_function_list + exp_bm)
        y_all = np.asarray(ltc_function_list + exp_ltc)
        x_pad = 0.10 * (x_all.max() - x_all.min())
        y_pad = 0.12 * (y_all.max() - y_all.min())
        ax.set_xlim(x_all.min() - x_pad, x_all.max() + x_pad)
        ax.set_ylim(y_all.min() - y_pad, y_all.max() + y_pad)
        ax_top.set_xlim(ax.get_xlim())
        ax_right.set_ylim(ax.get_ylim())

        #plt.xlabel("Predicted Bulk Modulus / GPa")
        ax.set_xlabel("Predicted Bulk Modulus [GPa]")
        #plt.ylabel("Predicted Lattice Thermal Conductivity / W·m⁻¹·K⁻¹")
        #plt.ylabel("Predicted Lattice Thermal Conductivity / $\mathrm{W \cdot m^{-1} \cdot K^{-1}}$")
        ax.set_ylabel("Predicted Lattice Thermal Conductivity [$\mathrm{W \cdot m^{-1} \cdot K^{-1}}$]", fontsize=20)
        if subspace == "FCC":
            fig.suptitle("Quinary FCC — marginal distribution of Os and Ir", fontsize=22, y=0.985)
        elif subspace == "BCC":
            fig.suptitle("Quinary BCC — marginal distribution of Mo and Ti", fontsize=22, y=0.985)
        fig.savefig(os.path.join(out_fig_dir, f"fig_marginal_{subspace}_{'-'.join(TARGET_ELEMENTS[subspace])}.png"),
                    dpi=600, bbox_inches='tight', pad_inches=0.35)
        plt.close(fig)

        system_choose = []
        for i in range(len(element_list)):
            system = []
            system.append(element_list[i])
            system.append(bm_function_list[i])
            system.append(ltc_function_list[i])
            system_choose.append(system)

        result_list = []
        for i in range(len(ratio_list)):
            result = []
            composition = system_choose[i][0][0] + str(int(108*ratio_list[i][0])) + "-" + system_choose[i][0][1] + str(int(108*ratio_list[i][1])) + "-" + system_choose[i][0][2] + str(int(108*ratio_list[i][2]))+ "-" + system_choose[i][0][3] + str(int(108*ratio_list[i][3]))+ "-" + system_choose[i][0][4] + str(int(108*ratio_list[i][4]))
            poscar_name = structure_file_list[i]
            bm_pred = system_choose[i][1]
            ltc_pred = system_choose[i][2]
            result.append(composition)
            result.append(poscar_name)
            result.append(bm_pred)
            result.append(ltc_pred)
            result_list.append(result)

        df = pd.DataFrame(data = result_list, columns=["composition", "poscar_name", "bm_pred_GPa", "ltc_pred_WmK"])
        for ele in TARGET_ELEMENTS[subspace]:
            df[f"frac_{ele}"] = target_frac[ele]
        df.to_csv(os.path.join(out_data_dir, f'{subspace}_alloys_dedup_mapped.csv'), index=False, encoding='utf-8-sig')
        print(subspace, "deduped alloys:", len(structure_file_list), "-> fig + csv saved")
