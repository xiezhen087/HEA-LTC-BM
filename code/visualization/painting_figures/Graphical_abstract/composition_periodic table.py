# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 14:57:53 2025

@author: Yaowei Wang.

Refer to https://ywwang0.github.io/2020/11/09/python%E7%BB%98%E5%88%B6%E5%85%83%E7%B4%A0%E5%91%A8%E6%9C%9F%E8%A1%A8/

"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
# import seaborn as sns
from pymatgen.core.periodic_table import Element
# sns.set(color_codes=True)

class periodic_table(object):
    """
    frame: str or dict, describe shapes of element cells.
        str: patch class name under matplotlib.patches.
        dict: {
            "shape": patch class name,
            "parms": width, height and other parms, needed for "shape",
            "kparms": kparms, supported by "shape"
        }
    colors: dict, custom frame and background colors for elements.
        {
            "frame": str, list
            "background": str, list, {"depend_on": data_key [, "cmp": color map, "cbar": True]}
        }
        if you use a gradient, color will based on the data from Element().data or you provide.
        depend_on specify which data you want to use.
    label: dict or function(not supported yet).
        dict: {
            "label1": [element symbols or atomic indexes],
            ...
        }
        or more detail:
        dict: {
            "label1": {
                "elements": [element symbols or atomic indexes],
                "color": not necessary
            }
            ...
        }
        function: func(element symbol or atomic index) returns color and label
    text: text content and format within element cells (not supported yet).
    data: data beside pymatgen or you want overide for pymatgen
        {symbol: {k1: v1, k2:v2}, ...}
    """

    def __init__(self,
                 frame=None,
                 colors=None,
                 labels=None,
                 text=None,
                 data=None):
        self.frame = frame or {}
        if isinstance(self.frame, str):
            self.frame = {"shape": self.frame or None}
        self.frame.setdefault("shape", "Rectangle")
        self.frame.setdefault("parms_orig", self.frame.get("parms", [1.0, 1.0]))
        self.frame.setdefault("kparms", {})
        self.colors = colors or {}
        self.colors.setdefault("frame", "black")
        self.colors.setdefault("background", "none")
        if (not labels) or "background" in labels or "frame" in labels:
            self.labels = labels or {}
        else:
            self.labels = {"background": labels}
        self._text = text or None
        self._data = {
            Element.from_Z(i).symbol: {"Symbol": Element.from_Z(i).symbol, **Element.from_Z(i).data}
            for i in range(1, 104)
        }
        if data:
            [self._data[k].update(data[k]) for k in data]

    def get_frame(self):
        self.frame["parms"] = list(self.frame["parms_orig"])
        if self.frame["shape"] == "FancyBboxPatch":
            self.frame["kparms"].setdefault("boxstyle", "round, pad=0.1")
            pad = float([*re.findall("pad=(\d+[.]\d+)", self.frame["kparms"]['boxstyle']), 0.3][0])
            self.frame["parms"][0:2] = list(np.array(self.frame["parms_orig"][0:2]) - 2*pad)
        try:
            return getattr(mpatches, self.frame["shape"])
        except AttributeError:
            raise ValueError("Unsupported shape error")

    def get_xys(self):
        width_and_height = np.array(self.frame["parms_orig"][0:2])
        text_shift = self.frame.get("text_shift",
                                    np.array([0.5, 0.5]) * width_and_height)
        if self.frame["shape"] == "FancyBboxPatch":
            text_shift -= 0.10 * width_and_height

        elements = [Element.from_Z(i) for i in range(1, 104)]
        group_and_rows = [np.array([el.group, el.row]) for el in elements]
        frame_xys = [
            np.array([1, -1]) * gr * width_and_height for gr in group_and_rows
        ]
        text_xys = [text_shift + fxy for fxy in frame_xys]
        
        #La
        for i in range(56,71):
            frame_xys[i] = frame_xys[i] + np.array([1.0 * (i- 56), -2.5])
            text_xys[i] = text_xys[i] + np.array([1.0 * (i- 56), -2.5])
        #Ac
        for i in range(88,103):
            frame_xys[i] = frame_xys[i] + np.array([1.0 * (i- 88), -2.5])
            text_xys[i] = text_xys[i] + np.array([1.0 * (i- 88), -2.5])
            
        return frame_xys, text_xys

    def get_colors_and_legend(self):
        for v in self.colors.values():
            if isinstance(v, list) and len(v) != len(self._data):
                raise ValueError(
                    "Length of color list should be compatible with Elements.")
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#7890af','#CB9475','#8CBF87','#af6d66','#b8a3b5','#b1b3b2'])#change colors cycle
        default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        default_edgecolor = ["black", self.colors["frame"]
                             ][isinstance(self.colors["frame"], str)]
        default_facecolor = ["none", self.colors["background"]
                             ][isinstance(self.colors["background"], str)]

        colors = {
            "edgecolors":
            self.colors.get("frame", default_edgecolor),
            "facecolors":
            [self.colors.get("background", default_facecolor),
             None][isinstance(self.colors["background"], dict)]
        }
        legend_handles = []

        for k, v in colors.items():
            labels = self.labels.get({
                "edgecolors": "frame",
                "facecolors": "background"
            }[k], None)
            if labels:
                if not v:
                    raise ValueError(
                        "Gradient mapped background can't be labelled.")
                if isinstance(colors[k], str):
                    colors[k] = [colors[k]] * len(self._data)
                if isinstance(list(labels.values())[0], list):
                    labels = {
                        lk: {
                            "elements": lv
                        }
                        for lk, lv in labels.items()
                    }
                for i, l in enumerate(labels):
                    label = labels[l]
                    label.setdefault("label", l)
                    indexes = [Element(s).Z - 1 for s in label["elements"]]
                    label.setdefault("color", default_color_cycle[i])
                    for j in indexes:
                        colors[k][j] = label["color"]
                    label.setdefault("frame_kparms", {})
                    for lkk in ["edgecolor", "facecolor"]:
                        label["frame_kparms"].setdefault(lkk, "none")
                    label["frame_kparms"][k[0:-1]] = label["color"]

                    patch_legend = mpatches.Patch(
                        label=label["label"], **label["frame_kparms"])
                    legend_handles.append(patch_legend)

        return colors, legend_handles

    def get_plot(self, title=None, figsize=None, legend_kparms={}):
        fig, ax = plt.subplots(figsize=figsize)

        frame_xys, text_xys = self.get_xys()
        colors, legend_handles = self.get_colors_and_legend()

        frame = self.get_frame()
        frames = [
            frame(xy, *self.frame["parms"], **self.frame["kparms"])
            for xy in frame_xys
        ]
        
        special_elements = {
            'Ti': {'hatch_color': '#af6d66', 'facecolor': '#7890af'},
            'V': {'hatch_color':'#8CBF87','facecolor': '#7890af'}
        }
        
        ax.add_collection(PatchCollection(frames, **colors))
        
        """
        for el_symbol, el_data in special_elements.items():
            el_index = Element(el_symbol).Z - 1
            el_xy = frame_xys[el_index]
            el_patch = frame(el_xy, *self.frame["parms"], edgecolor=el_data['hatch_color'], facecolor=el_data['facecolor'], hatch='//', **self.frame["kparms"])
            ax.add_patch(el_patch)  
        for el_symbol, el_data in special_elements.items():
            el_index = Element(el_symbol).Z - 1
            el_xy = frame_xys[el_index]
            el_patch = frame(el_xy, *self.frame["parms"], edgecolor=colors["edgecolors"], facecolor= None, **self.frame["kparms"])
            ax.add_patch(el_patch)
        """
        
        
        [
            ax.text(
                *text_xys[i],
                Element.from_Z(i + 1).symbol,
                horizontalalignment="center",
                verticalalignment="center",
                family="Times New Roman",
                size="x-large"
            ) for i in range(len(self._data)) if i not in [21, 22]  # Skip Sc/Ti elements
        ]


        if legend_handles:
            legend_kparms.update({"handles": legend_handles})
            legend_kparms.setdefault("loc", "upper center")
            legend_kparms.setdefault("bbox_to_anchor", (0.41, 1.0))
            legend_kparms.setdefault("fontsize", "x-large")
            ax.legend(**legend_kparms)
        if title:
            plt.title(title, fontsize=28)
        plt.legend()
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        return plt

    def show(self, title=None, figsize=None):
        plt = self.get_plot(title=title, figsize=figsize)
        plt.savefig('./data/Element_simple.png',dpi = 600, bbox_inches='tight')
        plt.show()
#Usage example 1
p = periodic_table(
    frame={
        "shape": "Rectangle",
        "parms": [1.0, 1.25]
    },
    labels={
        "background": {
            "3d Metals": ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co','Ni', 'Cu', 'Zn'],
            "Refractory Metals": ['W', 'Ta', 'Mo', 'Nb', 'Zr', 'Hf', 'Re'],
            "Lightweight Metals" : ["Al", "Mg", "Be",  "K", "Na", "Sr", "Ba", "Ca", "Li"],
            "Low-melting-point Metals" : ["Sn","In",'Bi','Pb'],
            "Noble Metals": ['Au', 'Ag', 'Pt', 'Pd', 'Ir' ,'Os', 'Rh', 'Ru'],
            "Rare-earth Matals" : [ 'Sc', 'Y', 'La','Ce', 'Pr','Nd', 'Pm', 'Sm','Eu','Gd','Tb','Dy', 'Ho','Er', 'Tm','Yb', 'Lu'],
        }
    })
pplt = p.get_plot(figsize=(9, 5))
pplt.show()

#Usage example 2
"""  
plt.close()
p = periodic_table(
    frame={
        "shape": "FancyBboxPatch",
        "parms": [1.0, 1.25]
    },
    labels={
        "background": {
            "+1 Elements": ['Na', 'K', 'Rb', 'Ag', 'Cs', 'Tl', 'Cu'],
            "+3 Elements": [
                'Al', 'Sc', 'Cr', 'Fe', 'Co', 'Ga', 'Y', 'Ru', 'Rh', 'In',
                'Sb', 'La', 'Ce', 'Gd', 'Ir', 'Bi'
            ],
            "-2 Elements": ['O', 'S', 'Se', 'Te']
        }
    })
pplt = p.get_plot(figsize=(9, 5))
pplt.savefig("periodic_table.pdf", transparent=True)
pplt.show()
"""