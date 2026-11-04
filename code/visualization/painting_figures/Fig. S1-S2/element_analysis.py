# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:38:48 2024

@author: Realunknown

The codes in function painting_element_heat_map refer to http://gjcheng.fun/2021/09/28/Python%E7%BB%98%E5%9B%BE%E4%B9%8BMatplotlib%E7%BB%98%E5%88%B6%E5%85%83%E7%B4%A0%E5%91%A8%E6%9C%9F%E8%A1%A8%E7%83%AD%E5%8A%9B%E5%9B%BE/ (author: G.J.Cheng)

"""
from pymatgen.core.periodic_table import Element
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
#import mendeleev
from matplotlib.font_manager import FontProperties

metal_elements = ["Li","Na","K","Rb","Cs","Fr","Be","Mg","Ca","Sr","Ba","Ra","Sc","Y","Ti","Zr","Hf","Rf","V","Nb","Ta","Db","Cr","Mo","W","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh","Ir","Mt","Ni","Pb","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Ge","Sn","Sb","Bi","Mc","Po","Lv","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"]
metal_elements_no_radiation = ["Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba","Sc","Y","Ti","Zr","Hf","V","Nb","Ta","Cr","Mo","W","Mn","Tc","Re","Fe","Ru","Os","Co","Rh","Ir","Ni","Pb","Pt","Cu","Ag","Au","Zn","Cd","Hg","Al","Ga","In","Tl","Ge","Sn","Sb","Bi","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"]
gas_elements = ["H" ,"He", "N", "O", "F", "Ne", "Cl", "Ar", "Kr","Xe","Rn"]
halogens_elements = ["F", "Cl", "Br", "I", "At", "Ts"]
semiconductor_elements = ["Si","Ga", "As", "N", "In", "P", "Zn", "Cd", "S", "Se", "C", "Ge","Te", "Al", "B" ,"Sn"]
salt_elements = ["C","H","O","N","P","S","B","Si"]

def painting_element_components(element_count,data_type):
    if data_type == "Thermal Conductivity":
        data_name = "Lattice Thermal Conductivity"
    elif data_type == "Bulk Modulus":
        data_name = "Mechanical Properties"
    element_list = list(set(element_count))
    element_name = [Element.from_Z(ele).symbol for ele in element_list]
    count_num = [element_count.count(ele) for ele in element_list]
    element_count_result = zip(element_name,count_num)
    sorted_element_count = sorted(element_count_result,key=lambda x:-x[1])
    result = zip(*sorted_element_count)
    element_name, count_num = [list(x) for x in result]

    
    element_count_metal = []
    for ele in element_count:
        if Element.from_Z(ele).symbol in metal_elements_no_radiation:
            element_count_metal.append(ele)
    element_list_metal = list(set(element_count_metal))
    element_name_metal = [Element.from_Z(ele).symbol for ele in element_list_metal]
    count_num_metal = [element_count_metal.count(ele) for ele in element_list_metal]
    element_count_result_metal = zip(element_name_metal,count_num_metal)
    sorted_element_count_metal = sorted(element_count_result_metal,key=lambda x:-x[1])
    result_metal = zip(*sorted_element_count_metal)
    element_name_metal, count_num_metal = [list(x) for x in result_metal]
    
    #plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Arial"
    fig, (ax1, ax2) = plt.subplots(2, 1,dpi =600,figsize=(28, 8))
    fig.subplots_adjust(hspace=0.4)
    #data_type_list = data_type.split()
    if data_type == "Thermal Conductivity":
        fig.suptitle('Elemental Composition of AFLOW\'s ' + data_name + ' Data',fontsize = 30)
    elif data_type == "Bulk Modulus":
        fig.suptitle('Elemental Composition of Materials Project\'s ' + data_name + ' Data',fontsize = 30)

    rects1 = ax1.bar(element_name_metal, count_num_metal,color="orange")
    for rect in rects1:
        height = rect.get_height()
        #ax1.text(rect.get_x(), 1.02*height, '%s' %
        #        int(height), size=12, family="Times new roman")
        ax1.text(rect.get_x(), 1.02*height, '%s' %
                int(height), size=12, family="Arial")
    x1_label = ax1.get_xticklabels() 
    [x1_label_temp.set_fontsize(15) for x1_label_temp in x1_label]
    y1_label = ax1.get_yticklabels() 
    [y1_label_temp.set_fontsize(15) for y1_label_temp in y1_label]
    for tick in ax1.get_xticklabels():
        tick.set_rotation(-25)

    if data_type == "Thermal Conductivity":
        ax1.set_ylim(0,450)
    elif data_type == "Bulk Modulus":
        ax1.set_ylim(0,1200)
    
    ax1.set_xlabel('Metallic Elements (Non-radioactive)',fontsize = 15)
    ax1.set_ylabel('Sample Counts',fontsize = 15)

    rects2 = ax2.bar(element_name, count_num)
    for rect in rects2:
        height = rect.get_height()
        #ax2.text(rect.get_x(), 1.02*height, '%s' %
        #        int(height), size=10, family="Times new roman")
        ax2.text(rect.get_x(), 1.02*height, '%s' %
                int(height), size=10, family="Arial")
    x1_label = ax2.get_xticklabels() 
    [x1_label_temp.set_fontsize(12) for x1_label_temp in x1_label]
    y1_label = ax2.get_yticklabels() 
    [y1_label_temp.set_fontsize(12) for y1_label_temp in y1_label]
    for tick in ax2.get_xticklabels():
        tick.set_rotation(-25)
    
    if data_type == "Thermal Conductivity":
        ax2.set_ylim(0,750)
    elif data_type == "Bulk Modulus":
        ax2.set_ylim(0,2500)
    ax2.set_xlabel('All Elements',fontsize = 15)
    ax2.set_ylabel('Sample Counts',fontsize = 15)
    
    if data_type == "Thermal Conductivity":
        plt.savefig('C:/Users/PC/Desktop/Elements-LTC.png',dpi = 600)
    elif data_type == "Bulk Modulus":
        plt.savefig('C:/Users/PC/Desktop/Elements-BM.png',dpi = 600)
    plt.show()
    return 0

def painting_element_num(element_num_count,data_type):
    element_num = list(set(element_num_count))
    counts = []
    for ele in element_num:
        counts.append(element_num_count.count(ele))
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(aspect="equal"))

    # Merge wedges below the threshold into a single group
    if data_type == "Bulk Modulus": 
        data = counts
        ingredients = element_num[:-2]
        # Set the threshold
        threshold = 30
        data = [d if d >= threshold else 0 for d in data]
        data = data[:-2] + [sum([d for d in counts if d < threshold])]
        ingredients.append('5 and 6')
    elif data_type == "Thermal Conductivity":
        data = counts
        ingredients = element_num
    
    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return f"{pct:.1f}%\n({absolute:d})"
    
    colors = ['#3E608D','#CB9475','#8CBF87','#8D2F25','#4E1945','#909291']
    wedges, texts, autotexts = ax.pie(data, colors=colors, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))
    
    if data_type == "Bulk Modulus": 
        for i, text in enumerate(texts):
            if i == 1 or i == 2:
                continue
            x = autotexts[i]._x  # Get the x coordinate of each wedge
            y = autotexts[i]._y  # Get the y coordinate of each wedge
            if i == 0:
                autotexts[i].set_y(y * 5.1)
            elif i == 3:
                autotexts[i].set_y(y * 3.6)
            elif i == 4:
                autotexts[i].set_text("< 0.3%")
                autotexts[i].set_x(x+0.13)
                autotexts[i].set_y(y-0.02)
    elif data_type == "Thermal Conductivity":
        pass
    
    ax.legend(wedges, ingredients,
              title="Number of Elements",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.setp(autotexts, size=12, weight="bold")
    
    if data_type == "Thermal Conductivity":
        ax.set_title("Element Numbers in AFLOW")
    elif data_type == "Bulk Modulus":
        ax.set_title("Element Numbers in Materials Project")
    
    fig.tight_layout(pad=2)
    
    if data_type == "Thermal Conductivity":
        plt.savefig('./Elements_numbers-LTC.png',dpi = 600)
    elif data_type == "Bulk Modulus":
        plt.savefig('./Elements_numbers-BM.png',dpi = 600)
    
    plt.show()
    return 0

#def if_

def painting_materials_type(materials_type_count,data_type):
    #plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Arial"
    num_materials_types = []
    materials_types = list(set(materials_type_count))
    for types in materials_types:
        num_materials_types.append(materials_type_count.count(types))
    
    bars = plt.bar(materials_types, num_materials_types, color = "#3E608D")
    for rect in bars:
        height = rect.get_height()
        #plt.text(rect.get_x() + rect.get_width()/2., 1.02*height, '%s' %
        #        int(height), size=18, family="Times new roman",ha='center')
        plt.text(rect.get_x() + rect.get_width()/2., 1.02*height, '%s' %
                int(height), size=18, family="Arial",ha='center')
    
    plt.xticks(rotation=45,fontsize = 14)
    plt.xlabel('Materials Types',fontsize = 18)
    plt.ylabel('Frequency',fontsize = 18)
    if data_type == "Thermal Conductivity":
        plt.ylim(0,3000)
    elif data_type == "Bulk Modulus":
        plt.ylim(0,6500)    
    
    if data_type == "Thermal Conductivity":
        plt.title('Material Types in AFLOW',fontsize = 20)
    elif data_type == "Bulk Modulus":
        plt.title('Material Types in Materials Project',fontsize = 20)
    
    if data_type == "Thermal Conductivity":
        #plt.savefig('C:/Users/PC/Desktop/Materials_types-LTC.png',dpi = 600, bbox_inches='tight')
        plt.savefig('./Materials_types-LTC.png',dpi = 600, bbox_inches='tight')
    elif data_type == "Bulk Modulus":
        #plt.savefig('C:/Users/PC/Desktop/Materials_types-BM.png',dpi = 600, bbox_inches='tight')
        plt.savefig('./Materials_types-BM.png',dpi = 600, bbox_inches='tight')
    
    plt.show()
    return 0

def materials_type(element_count_sample):
    metal_element_count = 0
    oxygen_element_count = 0
    halogens_element_count = 0
    gas_element_count = 0
    semiconductor_element_count = 0
    salt_element_count = 0
    for ele in element_count_sample:
        if Element.from_Z(ele).symbol in metal_elements:
            metal_element_count += 1
        if Element.from_Z(ele).symbol == "O":
            oxygen_element_count += 1
        if Element.from_Z(ele).symbol in halogens_elements:
            halogens_element_count += 1
        if Element.from_Z(ele).symbol in gas_elements:
            gas_element_count += 1
        if  Element.from_Z(ele).symbol in semiconductor_elements:
            semiconductor_element_count += 1
        if  Element.from_Z(ele).symbol in salt_elements:
            salt_element_count += 1
            
    if metal_element_count == len(element_count_sample):
        return "Metals and Alloys"
    elif metal_element_count + oxygen_element_count == len(element_count_sample):
        return "Metallic Oxide"
    elif metal_element_count + halogens_element_count == len(element_count_sample):
        return "Metal Halide"
    elif gas_element_count == len(element_count_sample):
        return "Gas"
    elif salt_element_count + metal_element_count == len(element_count_sample):
        return "Common Salt"
    elif semiconductor_element_count > len(element_count_sample)*0.7:
        return "Semiconductor-like"
    else:
        return "Others"
    
def painting_element_components_stacked(element_count, element_count_metal_metal,data_type):
    element_count_metal = []
    for ele in element_count:
        if Element.from_Z(ele).symbol in metal_elements_no_radiation:
            element_count_metal.append(ele)
    element_list_metal = list(set(element_count_metal))
    element_name_metal = [Element.from_Z(ele).symbol for ele in element_list_metal]
    count_num_metal = [element_count_metal.count(ele) for ele in element_list_metal]
    element_count_result_metal = zip(element_name_metal,count_num_metal)
    sorted_element_count_metal = sorted(element_count_result_metal,key=lambda x:-x[1])
    result_metal = zip(*sorted_element_count_metal)
    element_name_metal, count_num_metal = [list(x) for x in result_metal]
    count_num_metal_metal = [element_count_metal_metal.count(Element(ele).Z) for ele in element_name_metal]
    
    count_num_metal_compound = [count_num_metal[i] - count_num_metal_metal[i] for i in range(len(element_name_metal))]
    data = {
    'num_metal_in_compound':count_num_metal_compound,
    'num_metal_in_metal': count_num_metal_metal,
    'Element': element_name_metal
    }
    
    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    
    #plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Arial"
    # Set the figure size
    plt.figure(figsize=(24, 8), dpi = 600)  # Set width to 12 inches and height to 7 inches
    
    # Initialize the bottom array
    bottom = np.zeros(len(df['Element']))
    
    # Draw the stacked bar chart
    plt.bar(x=df['Element'], height=df['num_metal_in_compound'], bottom=bottom, color = "#3E608D" , label= 'Metallic elements in compounds')
    bottom += df['num_metal_in_compound']  # Update the bottom
    
    plt.bar(x=df['Element'], height=df['num_metal_in_metal'], bottom=bottom,color = "#CB9475" , hatch='/',edgecolor = "white", label='Metallic elements in pure metals or alloys')
    bottom += df['num_metal_in_metal']  # Update the bottom
    
    # Set the chart title and labels
    if data_type == "Thermal Conductivity":
        plt.title('Metallic Elements Ocurrence Counts in AFLOW', fontsize=30)  # Adjust the title font size and weight
    elif data_type == "Bulk Modulus":
        plt.title('Metallic Elements Ocurrence Counts in Materials Project', fontsize=30)  # Adjust the title font size and weight
    plt.xlabel("Metallic Elements (Non-radioactive)", fontsize=24)  # Adjust the x-axis label font size
    plt.ylabel('Sample Counts', fontsize=24)  # Adjust the y-axis label font size
    
    # Show the legend
    plt.legend(fontsize=24)
    plt.xticks(fontsize=18)
    
    # Adjust the rotation angle of the x-axis tick labels to avoid overlap
    plt.xticks(rotation= -25)
    """
    if data_type == "Thermal Conductivity":
        plt.savefig('C:/Users/PC/Desktop/Matel_types-LTC.png',dpi = 600)
    elif data_type == "Bulk Modulus":
        plt.savefig('C:/Users/PC/Desktop/Matel_types-BM.png',dpi = 600)
    """
    # Display the chart
    plt.show()
    return 0

def painting_element_heat_map(element_count,data_type):
    if data_type == "Thermal Conductivity":
        data_name = "Lattice Thermal Conductivity"
    elif data_type == "Bulk Modulus":
        data_name = "Mechanical Properties"
    element_list = list(set(element_count))
    element_name = [Element.from_Z(ele).symbol for ele in element_list]
    count_num = [element_count.count(ele) for ele in element_list]
    element_count_result = zip(element_name,count_num)
    sorted_element_count = sorted(element_count_result,key=lambda x:-x[1])
    result = zip(*sorted_element_count)
    element_name, count_num = [list(x) for x in result]
    
    plot_data = {k: v for k, v in zip(element_name, count_num)}
    
    # Cell settings in the periodic table
    # Cell size
    cell_length = 1
    # Gap between cells
    cell_gap = 0.1
    # Cell border width
    cell_edge_width = 0.5
    
    # Get each element's atomic number, period (row), group (column), and plot data (set to 0 if missing)
    elements = []
    for i in range(1, 119):
        ele = mendeleev.element(i)
        ele_group, ele_period = ele.group_id, ele.period
    
        # Place the lanthanides in row 8
        if 57 <= i <= 71:
            ele_group = i - 57 + 3
            ele_period = 8
        # Place the actinides in row 9
        if 89 <= i <= 103:
            ele_group = i - 89 + 3
            ele_period = 9
    
        elements.append([i, ele.symbol, ele_group, ele_period,
                         plot_data.setdefault(ele.symbol, 0)])
    
    # Set the annotation labels for the La and Ac series
    elements.append([None, 'LA', 3, 6, None])
    elements.append([None, 'AC', 3, 7, None])
    elements.append([None, 'LA', 2, 8, None])
    elements.append([None, 'AC', 2, 9, None])
    
    # Create a new Matplotlib figure window
    fig, ax = plt.subplots(figsize=(10, 5))
    # Range of the x and y axes
    xy_length = (20, 11)
    
    # Get the YlOrRd colormap
    #my_cmap = cm.get_cmap('YlOrRd')
    my_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'custom_red', 
    [(0, '#FFFFFF'), (1, '#8D2F25')],  # white → dark red
    N=256
)
    # Map the plot_data values to colors; adjust according to the actual situation
    if data_type == "Thermal Conductivity": 
        norm = mpl.colors.Normalize(1, 500)
    elif data_type == "Bulk Modulus":
        norm = mpl.colors.Normalize(1, 1000)
    # Set the color below the colormap's lower limit (None means unset, i.e. white)
    my_cmap.set_under('None')
    # Link the colorbar and the mapping
    cmmapable = cm.ScalarMappable(norm, my_cmap)
    # Draw the colorbar
    plt.colorbar(cmmapable, ax=ax, drawedges=False)
    
    # Draw the periodic table cells and fill in the attributes and colors
    for e in elements:
        ele_number, ele_symbol, ele_group, ele_period, ele_count = e
        print(ele_number, ele_symbol, ele_group, ele_period, ele_count)
    
        if ele_group is None:
            continue
    
        # x, y locate the cell position
        x = (cell_length + cell_gap) * (ele_group - 1)
        y = xy_length[1] - ((cell_length + cell_gap) * ele_period)
    
        # Increase the distance of the La and Ac series from the periodic table
        if ele_period >= 8:
            y -= cell_length * 0.5
    
        # Draw the cell border and fill it with the heat color when the cell's atomic number is not None
        # i.e., do not draw the border or color fill for the La and Ac annotation labels
        if ele_number:
            fill_color = my_cmap(norm(ele_count))
            rect = patches.Rectangle(xy=(x, y),
                                     width=cell_length, height=cell_length,
                                     linewidth=cell_edge_width,
                                     edgecolor='k',
                                     facecolor=fill_color)
            plt.gca().add_patch(rect)
    
        # Add the atomic number attribute in the cell
        plt.text(x + 0.04, y + 0.8,
                 ele_number,
                 va='center', ha='left',
                 fontdict={'size': 6, 'color': 'black', 'family': 'Helvetica'})
        # Add the element symbol in the cell
        plt.text(x + 0.5, y + 0.5,
                 ele_symbol,
                 va='center', ha='center',
                 fontdict={'size': 9, 'color': 'black', 'family': 'Helvetica', 'weight': 'bold'})
        # Add the heat value in the cell
        plt.text(x + 0.5, y + 0.12,
                 ele_count,
                 va='center', ha='center',
                 fontdict={'size': 6, 'color': 'black', 'family': 'Helvetica'})
    
    # Set the x and y axes to equal scale (1:1) (so the cells look square)
    plt.axis('equal')
    # Turn off the axes
    plt.axis('off')
    # Trim the blank margins
    plt.tight_layout()
    # Set the range of the x and y axes
    plt.ylim(0, xy_length[1])
    plt.xlim(0, xy_length[0])
    
    #font = FontProperties(family='serif', name='Times New Roman', size=20)
    if data_type == "Thermal Conductivity":
        #plt.title("Element Frequency in AFLOW's LTC Data", fontdict={'family': 'Times New Roman', 'size': 20})
        plt.title("Element Frequency in AFLOW's LTC Data", fontdict={'family': 'Arial', 'size': 20})
    elif data_type == "Bulk Modulus":
        #plt.title("Element Frequency in Materials Project's Mechanical Property Data", fontdict={'family': 'Times New Roman', 'size': 20})
        plt.title("Element Frequency in Materials Project's Mechanical Property Data", fontdict={'family': 'Arial', 'size': 20})
    
    # Save the figure
    """
    if data_type == "Thermal Conductivity":
        plt.savefig('C:/Users/PC/Desktop/Element_heat_types-LTC.png',dpi = 600, bbox_inches='tight')
    elif data_type == "Bulk Modulus":
        plt.savefig('C:/Users/PC/Desktop/Element_hest_types-BM.png',dpi = 600, bbox_inches='tight')
    """
    # Show the figure window
    plt.show()
    return 0

def count(data_set,data_type):
    element_count = []
    element_num_count = []
    materials_type_count = []
    data_set_metal = []
    for data in data_set:
        element_count_sample = []
        for d in data.x:
            element_count_sample.append(int(d[0]))
        element_count_sample = list(set(element_count_sample))
        element_num_count.append(len(element_count_sample))
        m_type = materials_type(element_count_sample)
        materials_type_count.append(m_type)
        element_count = element_count + element_count_sample
        if m_type == "Metals and Alloys":
            data_set_metal.append(data)
    #painting_element_components(element_count,data_type)
    painting_element_num(element_num_count,data_type)
    painting_materials_type(materials_type_count,data_type)
    #painting_element_heat_map(element_count,data_type)
    
    element_count_metal = []
    for data in data_set_metal:
        element_count_sample_metal = []
        for d in data.x:
            element_count_sample_metal.append(int(d[0]))
        element_count_sample_metal = list(set(element_count_sample_metal))
        element_count_metal = element_count_metal + element_count_sample_metal
    #painting_element_components_stacked(element_count, element_count_metal,data_type)
    return element_count, element_num_count

if __name__ == '__main__' :
    data_path = "../../trainning/data_set/"
    #data_types = ["Formation Energy","Bulk Modulus","Thermal Conductivity"]
    data_types = ["Bulk Modulus"]
    #data_types = ["Formation Energy"]
    for data_type in data_types:
        if data_type == "Formation Energy":
            #data_set = torch.load(data_path + "energy/cgcnn-voronoi-Formation_energy-metal-2.3w.pt", weights_only=False)
            data_set = torch.load(data_path + "energy/cgcnn-voronoi-Formation_energy-metal-2.3w.pt")
            element_count, element_num_count = count(data_set,data_type)
        elif data_type == "Bulk Modulus":
            #data_set = torch.load(data_path +  "Elasiticity/cgcnn-voronoi-bulk_modulus-HEA.pt", weights_only=False)
            data_set = torch.load(data_path +  "Elasiticity/bulk_modulus/cgcnn-voronoi-bulk_modulus-HEA.pt")
            element_count, element_num_count = count(data_set,data_type)
        elif data_type == "Thermal Conductivity":
            #data_set = torch.load(data_path + "thermal_conduct/cgcnn-voronoi-thermal_conductivity-HEA.pt", weights_only=False)
            data_set = torch.load(data_path + "thermal_conduct/cgcnn-voronoi-thermal_conductivity-HEA.pt")
            element_count, element_num_count = count(data_set,data_type)