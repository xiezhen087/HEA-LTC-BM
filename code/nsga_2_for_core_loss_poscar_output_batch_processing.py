"""
Main codes refer to:
    https://blog.csdn.net/no___good/article/details/134749475?type=blog&rId=134749475&refer=APP&source=mr_unknown23333
    @no__good

Our contributions: 
    1. On his basis, we extended the objective function to a nine-dimensional function.
    2. Plotted the log of Pareto Front.
    3. And plotted the HV diagram of the entire optimization process.
    
Thanks to the open source community:
    github.com
    gitee.com

And chinese website:
    csdn.com
    baidu.com
    
All dependencies on open source python packages are available in the file:
    requirement.txt

And the use of the code is available in the file:
    readme.md
    
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from pymoo.indicators.hv import HV
from mixmodel import MixModel
from torch_geometric.loader import DataLoader
from datetime import datetime
import sys
sys.path.append("./")
from data_combine import output_data
import torch
from pymatgen.core.periodic_table import Element
import itertools
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure, Lattice
import torch.nn as nn
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
class Individual(object):
    def __init__(self):
        self.solution = None
        self.objective = defaultdict()
 
        self.n = 0
        self.rank = 0
        self.S = []
        self.distance = 0
        self.Gen_cur = -1
 
    def bound_process(self, bound_min, bound_max):
        for i, item in enumerate(self.solution):
            if item > bound_max[i]:
                self.solution[i] = bound_max[i]
            elif item < bound_min[i]:
                self.solution[i] = bound_min[i]
    """
    def calculate_objective(self, objective_fun):
        self.objective = objective_fun(self.solution, self.Gen_cur)
    """
 
    def __lt__(self, other):
        v1 = list(self.objective.values())
        v2 = list(other.objective.values())
        for i in range(len(v1)):
            if v1[i] > v2[i]:# define the optimization direction (maximize or minimize)
                return 0 
        return 1

"""
def Generate_individual():
    x = [0]*9
    X_BOUND_0 = [-0.05, 0.07]  # time_skew value range
    X_BOUND_1 = [-2.0, -1.0]  # time_kurt value range
    X_BOUND_2 = [1.1, 1.7]  # time_form value range
    X_BOUND_3 = [0, 0.3]  # max_al value range
    X_BOUND_4 = [1.2, 2.2]  # time_impulse value range
    X_BOUND_5 = [1.5, 2.5]  # time_clearance value range
    X_BOUND_6 = [25, 50,70,90]  # temp value range
    X_BOUND_7 = [1,2,3,4]  # materials value range
    X_BOUND_8 = [0, 50000]  # freq value range
    x[0] = np.random.rand(1)[0] * (X_BOUND_0[1] - X_BOUND_0[0]) + X_BOUND_0[0]
    x[1] = np.random.rand(1)[0] * (X_BOUND_1[1] - X_BOUND_1[0]) + X_BOUND_1[0]
    x[2] = np.random.rand(1)[0] * (X_BOUND_2[1] - X_BOUND_2[0]) + X_BOUND_2[0]
    x[3] = np.random.rand(1)[0] * (X_BOUND_3[1] - X_BOUND_3[0]) + X_BOUND_3[0]
    x[4] = np.random.rand(1)[0] * (X_BOUND_4[1] - X_BOUND_4[0]) + X_BOUND_4[0]
    x[5] = np.random.rand(1)[0] * (X_BOUND_5[1] - X_BOUND_5[0]) + X_BOUND_5[0]
    x[6] = random.choice(X_BOUND_6)
    x[7] = random.choice(X_BOUND_7)
    x[8] = random.randint(X_BOUND_8[0], X_BOUND_8[1])
    #print("x_8_ini: {}".format(x[8]))
    return np.array(x)
"""

def modify_lattice_constant(structure,framework):
    lattice_constant = [0]*3
    species = structure.species
    num_species = len(set(species))
    species_type = list(set(species))
    species_radius_type = [ele.atomic_radius_calculated for ele in species_type]
    species_ratio = [species.count(ele)/len(species) for ele in species_type]
    if (num_species == 2) and framework == "fcc":
        lattice_constant[0] = 0.7069*sum(species_radius_type)+1.5677
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = lattice_constant[0]
    elif (num_species == 2) and framework == "bcc":
        lattice_constant[0] = 1.0884*sum(np.array(species_radius_type)*np.array(species_ratio))+1.2315
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = lattice_constant[0]
    elif (num_species == 2) and framework == "hcp":
        lattice_constant[0] = 1.3335*sum(species_radius_type)+1.2001
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = 8.2611*sum(np.array(species_radius_type)*np.array(species_ratio))-6.0775
    elif (num_species == 3) and framework == "fcc":
        lattice_constant[0] = 3.64570
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = lattice_constant[0]
    elif (num_species == 5) and framework == "fcc":
        lattice_constant[0] = 3.844382553
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = lattice_constant[0]
    elif (num_species == 5) and framework == "bcc":
        lattice_constant[0] = 3.239816519
        lattice_constant[1] = lattice_constant[0]
        lattice_constant[2] = lattice_constant[0]
    lattice = structure.lattice
    new_lattice = Lattice.from_parameters( a=lattice_constant[0]*3, b=lattice_constant[1]*3, c=lattice_constant[2]*3, alpha=lattice.alpha, beta=lattice.beta, gamma=lattice.gamma)
    new_structure = Structure(new_lattice, structure.species, structure.frac_coords)
    return new_structure

def re_construct_structure(structure ,element_list, ratio_list, framework):
    total_atom_num = len(structure)
    element_num = [int(total_atom_num * ratio) for ratio in ratio_list]
    element_num.append(int(total_atom_num - sum(element_num)))
    element_num = [0] + element_num 
    
    index_list = [i for i in range(total_atom_num)]
    random.shuffle(index_list)
    element_index_part = []
    for i in range(len(element_num) -1):
        ini_index = sum(element_num[0:i+1])
        end_index = sum(element_num[0:i + 2])
        element_index_part.append(index_list[ini_index:end_index])
    
    for i in range(len(element_index_part)):
        for j in range(len(element_index_part[i])):
           structure.replace(element_index_part[i][j], Element(element_list[i]))
    return structure

def input_poscar(element_list, ratio_list, framework):
    file_path = "../initial_structure/"
    num_species = len(element_list)
    if num_species == 2:
        if framework == "fcc":
            poscar = Poscar.from_file(file_path + "bi_alloy/fcc/" + "POSCAR_Fe_fcc",'r')
        elif framework == "bcc":
            poscar = Poscar.from_file(file_path + "bi_alloy/bcc/" + "POSCAR_W_bcc",'r')
        elif framework == "hcp":
            poscar = Poscar.from_file(file_path + "bi_alloy/hcp/" + "POSCAR_Zr_hcp",'r')
    elif num_species == 3:
        if framework == "fcc":
            poscar = Poscar.from_file(file_path + "ter_alloy/fcc/" + "POSCAR_Fe_fcc",'r')
        elif framework == "bcc":
            poscar = Poscar.from_file(file_path + "ter_alloy/bcc/" + "POSCAR_W_bcc",'r')
        elif framework == "hcp":
            poscar = Poscar.from_file(file_path + "ter_alloy/hcp/" + "POSCAR_Zr_hcp",'r')
    elif num_species == 5:
        if framework == "fcc":
            poscar = Poscar.from_file(file_path + "qui_alloy/fcc/" + "POSCAR_Fe_fcc",'r')
        elif framework == "bcc":
            poscar = Poscar.from_file(file_path + "qui_alloy/bcc/" + "POSCAR_W_bcc",'r')
        elif framework == "hcp":
            poscar = Poscar.from_file(file_path + "qui_alloy/hcp/" + "POSCAR_Zr_hcp",'r')
    structure = poscar.structure
    supercell = structure.make_supercell([3, 3, 3])
    structure = Structure(supercell.lattice, supercell.species, supercell.frac_coords)
    structure = re_construct_structure(structure ,element_list, ratio_list, framework)
    structure = modify_lattice_constant(structure,framework)
    return structure

def load_data(structure, material_property):
    data = output_data(structure,material_property)
    return data

def Generate_individual():
    x = [0]*(element_num + 1)
    
    x_element_composition = list(range(len(all_combinations)))  # value range of the element composition
    x_element_ratio = [0]*(element_num - 1)
    for i in range(element_num - 1):
        x_element_ratio[i] = [0.05, 1/element_num]  # fraction of the first element
    #x_framework = ["fcc","bcc","hcp"]  # freq value range
    x_framework = list(range(len(selected_framework)))  # framework value range
    
    x[0] = random.choice(x_element_composition)
    for i in range(element_num - 1):
        x[i + 1] = np.random.rand(1)[0] * (x_element_ratio[i][1] - x_element_ratio[i][0]) + x_element_ratio[i][0]
    x[-1] = random.choice(x_framework)
    #print(x)
    return np.array(x)

def individual_to_data(x_individual):
    element_num = len(x_individual) - 1
    ratio_list = [0]*(element_num - 1)
    
    element_list = all_combinations[int(x_individual[0])].split(" ")
    for i in range(element_num - 1):
        ratio_list[i] = float(x_individual[i + 1])
    #print(x_individual)
    framework = selected_framework[int(x_individual[-1])]
    structure = input_poscar(element_list, ratio_list, framework)
    
    data_bulk_modulus = load_data(structure,"bulk_modulus")
    data_thermal_conduct = load_data(structure,"thermal_conduct")
    data = (data_bulk_modulus,data_thermal_conduct)
    return data, structure

def give_value(P,value):
    #print(value)
    for i in range(len(P)):
        P[i].objective[1] = value[0][i]
        P[i].objective[2] = value[1][i]
    return P
 
def main():
    generations = 50# number of generations
    popnum = 250  # population size
    eta = 1  # mutation distribution parameter; the larger this value, the higher the probability that offspring approach their parents. Deb recommends setting it to 1
 
    # problem definition
    # poplength = 30  # dimension of an individual solution vector
    # bound_min = 0  # domain
    # bound_max = 1
    # objective_fun = ZDT1
 
    poplength = 6  # dimension of an individual solution vector
    #bound_min = [0,0.05,0.05,0.05,0.05,0]  # domain
    #bound_max = [len(all_combinations)-1,0.21,0.21,0.21,0.21,len(selected_framework)-1]
    bound_min = [0,0.05,0.05,0.05,0.05,0]  # domain
    bound_max = [len(all_combinations)-1,0.24,0.24,0.24,0.24,len(selected_framework)-1]
    #objective_fun = KUR  #  objective function
 
    # generate the first-generation population
    P = [] # create an empty list P to store individual objects
    gen_cur = -1
    print("-----------Begin!----------")
    print("Generate a population……")
    for i in range(popnum):
        print("Generate individual index:{}".format(i))
        # generate popnum individuals in a loop and add them to the list P.
        P.append(Individual())
        P[i].solution = Generate_individual()  # randomly generate a feasible solution for the individual
        #P[i].calculate_objective(objective_fun)  # calculate the objective function value
        P[i].Gen_cur = gen_cur
    fun_value_p = calculate_fun_value(P,gen_cur)
    P = give_value(P,fun_value_p)
    
    fast_non_dominated_sort(P)
    
    Q = make_new_pop(P, eta, bound_min, bound_max, gen_cur)
    fun_value_q = calculate_fun_value(Q,gen_cur)
    Q = give_value(Q,fun_value_q)
 
    P_t = P
    Q_t = Q
 
    for gen_cur in range(generations):
        if gen_cur > 10:
            os.mkdir("../searching/log/formal/" + save_file_name + "/" + save_file_name + "_" + save_path_name + "/" + str(gen_cur) + "/")
        
        R_t = P_t + Q_t
        F = fast_non_dominated_sort(R_t)
 
        P_n = []
        i = 1
        while len(P_n) + len(F[i]) < popnum:  
            crowding_distance_assignment(F[i])
            P_n = P_n + F[i] 
            i = i + 1
        F[i].sort(key=lambda x: x.distance) 
        P_n = P_n + F[i][:popnum - len(P_n)] 
        Q_n = make_new_pop(P_n, eta, bound_min, bound_max, gen_cur)
        fun_value_q = calculate_fun_value(Q_n,gen_cur)
        Q_n = give_value(Q_n,fun_value_q)
 

        P_t = P_n
        Q_t = Q_n
        plot_P(P_t,gen_cur + 1)
        plt.pause(0.1)
    plt.show()
 
    return 0
 
def fast_non_dominated_sort(P):
    F = defaultdict(list)
 
    for p in P:
        p.S = [] 
        p.n = 0 
        for q in P:
            if p < q:# dominance direction: choose whether to optimize toward the maximum or minimum objective value
                p.S.append(q)
            elif q < p:# dominance direction: choose whether to optimize toward the maximum or minimum objective value
                p.n += 1 
        if p.n == 0:
            p.rank = 1
            F[1].append(p)
    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i = i + 1
        F[i] = Q
 
    return F
 

def make_new_pop(P, eta, bound_min, bound_max,gen_cur):
    popnum = len(P)
    Q = []
    for i in range(int(popnum / 2)): 
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent1 = binary_tournament(P[i], P[j])
 
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent2 = binary_tournament(P[i], P[j])
 
        while (parent1.solution == parent2.solution).all():
            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent2 = binary_tournament(P[i], P[j])
 
        
        Two_offspring = crossover_mutation(parent1, parent2, eta, bound_min, bound_max, gen_cur)
 
        
        Q.append(Two_offspring[0])
        Q.append(Two_offspring[1])
    return Q
 

def crowding_distance_assignment(L):
    l = len(L) 
 
    for i in range(l):
        L[i].distance = 0
 
    for m in L[0].objective.keys():
        L.sort(key=lambda x: x.objective[m])
        L[0].distance = float('inf')
        L[l - 1].distance = float('inf')
 
        f_max = L[l - 1].objective[m]
        f_min = L[0].objective[m]
 
        try:
            for i in range(1, l - 1):
                L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)
        except Exception:
            print(str(m) + "Max Value: " + str(f_max) + "Min Value: " + str(f_min))
 
def binary_tournament(ind1, ind2):
    if ind1.rank != ind2.rank:
        return ind1 if ind1.rank < ind2.rank else ind2
    elif ind1.distance != ind2.distance:
        return ind1 if ind1.distance > ind2.distance else ind2
    else:
        return ind1
 

def crossover_mutation(parent1, parent2, eta, bound_min, bound_max, gen_cur):
    poplength = len(parent1.solution)
 
    offspring1 = Individual()
    offspring1.Gen_cur = gen_cur
    offspring2 = Individual()
    offspring2.Gen_cur = gen_cur
    offspring1.solution = np.empty(poplength)
    offspring2.solution = np.empty(poplength)
 
   
    for i in range(poplength):
        rand = random.random()
        beta = (rand * 2) ** (1 / (eta + 1)) if rand < 0.5 else (1 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
        offspring1.solution[i] = 0.5 * ((1 + beta) * parent1.solution[i] + (1 - beta) * parent2.solution[i])
        offspring1.bound_process(bound_min, bound_max)
        offspring2.solution[i] = 0.5 * ((1 - beta) * parent1.solution[i] + (1 + beta) * parent2.solution[i])
        offspring2.bound_process(bound_min, bound_max)
    for i in range(poplength):
        mu = random.random()
        delta = 2 * mu ** (1 / (eta + 1)) if mu < 0.5 else (1 - (2 * (1 - mu)) ** (1 / (eta + 1)))
        offspring1.solution[i] = offspring1.solution[i] + delta
        offspring1.bound_process(bound_min, bound_max)
    
    #offspring1.calculate_objective(objective_fun)
    #offspring2.calculate_objective(objective_fun)
 
    return [offspring1, offspring2]
 
"""
def goal_func_1(test_x):
    x = []
    x.append(test_x)
    x.append(test_x)
    x = np.array(x)
    clf = xgb.XGBRegressor()
    booster = xgb.Booster()
    booster.load_model('best_models/model_xgboost.bin')
    clf._Booster = booster
    model_xgb = clf
    min_value = 2.6186892424572115
    delta_value = 3.9395550971001576
    
    y_pred_x = model_xgb.predict(x)[0]
    y_real_x_1 = np.power(10,y_pred_x *delta_value + min_value)
    
    return np.log(y_real_x_1)
"""

def goal_func_bulk_modulus(data_list): #bulk_modulus
    model_path = "../pre_trained_models/bulk_modulus/result/0625_loop_save_time_1_r_square_0.8799/"
    
    model = MixModel()
    if torch.cuda.device_count() > 1:
        # if there is more than one GPU device, use two GPUs simultaneously
        model = nn.DataParallel(model, [0, 1])
    model=model.to(device)
    
    model.load_state_dict(torch.load(model_path + "model_mix_best_" + "bulk_modulus" + "_0625_except_pure_metal_loop_save_time_1.pth" ,map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load(model_path + "model_bo_temp-477-0.230.pt" ,map_location=torch.device('cpu')))
    normalization = torch.load(model_path + "normalization.pt")
    model.eval()
    
    val_loader = DataLoader(data_list, batch_size= len(data_list),shuffle= False)
    for batch in val_loader:
        for i in range(1,len(batch[0])):
            batch[0][i] = batch[0][i].to(device)
        batch[1].to(device)
        with torch.no_grad():
            energy_per_atom = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
    #print("Bulk Modulus:{}".format(energy_per_atom[0][0]))
    energy_per_atom_list = energy_per_atom.tolist()
    y_list = [float(energy_per_atom[i][0])*float(normalization[1]) + float(normalization[0]) for i in range(len(energy_per_atom_list))]
    #print("y:{}".format(y))
    return y_list

"""
def goal_func_2(test_x):
    y_real_x_2 = test_x[3]*test_x[8]
    return np.log(y_real_x_2)
"""
def goal_func_thermal_conduct(data_list):#lattice_thermal_conduct
    model_path = "../pre_trained_models/thermal_conduct/result/0802_loop_save_time_0_r_square_0.872/"
    
    model = MixModel()
    if torch.cuda.device_count() > 1:
        # if there is more than one GPU device, use two GPUs simultaneously
        model = nn.DataParallel(model, [0, 1])
    model=model.to(device)
    
    model.load_state_dict(torch.load(model_path + "model_mix_best_" + "thermal_conduct" + "_0802_except_pure_metal_loop_save_time_0.pth" ,map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load(model_path + "model_bo_temp-477-0.230.pt" ,map_location=torch.device('cpu')))
    normalization = torch.load(model_path + "normalization.pt")
    model.eval()
    
    val_loader = DataLoader(data_list, batch_size= len(data_list),shuffle= False)
    for batch in val_loader:
        for i in range(1,len(batch[0])):
            batch[0][i] = batch[0][i].to(device)
        batch[1].to(device)
        with torch.no_grad():
            energy_per_atom = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
    energy_per_atom_list = energy_per_atom.tolist()
    y_list = [float(energy_per_atom[i][0])*float(normalization[1]) + float(normalization[0]) for i in range(len(energy_per_atom_list))]
    #print("y:{}".format(y))
    return y_list

def pair_floats(float1, float2):
    return [float1,float2]

def calculate_hv(X,Y):
    ref_point = np.array([10, 0])
    ind = HV(ref_point=ref_point)
    A = []
    for i in range(len(X)):
        A_sin = []
        A_sin.append(X[i])
        A_sin.append(Y[i])
        A.append(A_sin)
    A = np.array(A)
    hv = ind(A)
    return hv

def calculate_fun_value(P, gen_cur):
    data_fun_1_list = []
    data_fun_2_list = []
    structure_list = []
    print("**************************")
    print("Structure generation begin!")
    now = datetime.now()
    print("Finishing Time:", now)
    for i in range(len(P)):
        print(i)
        data_tuple, structure = individual_to_data(P[i].solution)
        data_fun_1_list.append(data_tuple[0])
        data_fun_2_list.append(data_tuple[1])
        structure_list.append(structure)
    print("Structure generation finished!")
    now = datetime.now()
    print("Finishing Time:", now)
    
    f = [0,0]

    
    f_0 = goal_func_bulk_modulus(data_fun_1_list)#max
    f[0] = [ele * (-1) for ele in f_0]
    f_1 = goal_func_thermal_conduct(data_fun_2_list)
    f[1] = [ele * (-1) for ele in f_1]
    print("AI reasoning completed!")
    print("Finishing Time:", now)
    if gen_cur > 10:
        for i in range(len(structure_list)):
            elements = structure_list[i].species
            coords = structure_list[i].frac_coords
            sorted_indices = sorted(range(len(elements)), key=lambda k: elements[k])
            sorted_elements = [elements[l] for l in sorted_indices]
            sorted_coords = [coords[m] for m in sorted_indices]
            new_structure = Structure(structure_list[i].lattice, sorted_elements, sorted_coords)
            new_structure.to("../searching/log/formal/" + save_file_name + "/" + save_file_name + "_" + save_path_name + "/" + str(gen_cur) + "/" + "POSCAR_" + str(int(P[i].solution[0])) + '_{:.2f}'.format(f[0][i]*(-1)) + '_{:.2f}'.format(f[1][i]*(-1)),fmt = "poscar")
    print("POSCAR output completed!") 
    print("Finishing Time:", now)
    return f
 
def plot_P(P,gen_cur):
    X = []
    Y = []
    for ind in P:
        X.append(-ind.objective[1])
        Y.append(-ind.objective[2])
    
    plt.figure(dpi = 600)
    fig, ax = plt.subplots(figsize=(8,5),dpi = 600)
    ax.scatter(X, Y, s = 80, marker='o',edgecolors='darkgoldenrod',facecolors='gold', label='NSGA-2:Multi-Objective Optimization')
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.grid()
    plt.title("Pareto Front Plot Generation:{}".format(gen_cur))
    
    if gen_cur > 10:
        torch.save(P,"../searching/log/formal/" + save_file_name + "/" + save_file_name + "_" + save_path_name + "/" + save_file_name + "_pareto_front_indvidual_{}.pt".format(gen_cur))
        plt.savefig("../searching/log/formal/" + save_file_name + "/" + save_file_name + "_" + save_path_name + "/" + save_file_name + "_pareto_front_indvidual_{}.png".format(gen_cur))
        pic = [X,Y]
        torch.save(pic,"../searching/log/formal/" + save_file_name + "/" + save_file_name + "_" + save_path_name + "/" + save_file_name + "_plot_x_y_{}.pt".format(gen_cur))
    plt.show()
    print("numbers of indvidual:{}".format(len(P)))
    hv = calculate_hv(X,Y)
    hv_list.append(hv)

def painting_hv(hv_list):
    x = np.linspace(0,len(hv_list),len(hv_list))
    plt.figure(dpi = 600)
    plt.plot(x, hv_list, color='darkgoldenrod') 
    plt.grid()
    plt.title('Hypervolume with NSGA-II')
    plt.savefig("../searching/log/formal/" + save_file_name + "/" + save_file_name + "_" + save_path_name + "/250327_quinary_nsga_pareto_front_indvidual_HV_curve.png")
    plt.show()
    return 0

# Entering main function
if __name__ == '__main__':
    element_num = 5 #num of elements in high entropy alloy
    save_path_name_list = ["Fe_total","Fe_single","Fe_noble","Fe_mix","Zr_total","Zr_single","Zr_noble","Zr_rare","Zr_mix","Cu_total","Cu_single","Cu_mix","Mg_total","Mg_single","Mg_mix","Ti_total","Ti_single","Ti_noble","Ti_rare","Ti_mix"]
    #save_path_name_list = ["Ti_mix_2"]
    #save_path_name_list = ["Fe_total"]
    save_file_name = "250513_nsga_quinary_bcc_multi_goal"
    for save_path_name in save_path_name_list:
        try:
            # searching elements
            os.mkdir("../searching/log/formal/" + save_file_name + "/" + save_file_name + "_" + save_path_name)
            # searching elements(ternary)
            if save_path_name == "Fe_total":
                selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Ru","Co","Rh","Pd","Ni","Pt","Cu","Ag","Au","Zn","Ga","Sn","Sb","Bi","La"]#Fe_total
            elif save_path_name == "Fe_single":
                selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Sn","Sb","Bi"]#Fe_single
            elif save_path_name == "Fe_noble":
                selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Ru","Co","Rh","Pd","Ni","Pt","Cu","Ag","Au","Zn","Ga","Sn","Sb","Bi"]#Fe_noble
            elif save_path_name == "Fe_mix":
                selected_element_list = ["Li","Mg","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Sn","Sb","Bi","Sc","Tc","Ir","Mo","W"]#Fe_mix
            elif save_path_name == "Zr_total":
                selected_element_list = ["Ca","Sr","Sc","Y","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","Ru","Os","Ir","Pt","Nd","Pm","Eu","Gd","Tb","Dy","Er","Tm","Yb","Lu"]#Zr_total
            elif save_path_name == "Zr_single":
                selected_element_list = ["Ca","Sr","Ti","Zr","Hf","Nb","Ta","Mo","W","Re"]#Zr_single
            elif save_path_name == "Zr_noble":
                selected_element_list = ["Ca","Sr","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","Ru","Os","Ir","Pt"]#Zr_noble
            elif save_path_name == "Zr_rare":
                selected_element_list = ["Ca","Sr","Sc","Y","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","Nd","Pm","Eu","Gd","Tb","Dy","Er","Tm","Yb","Lu"]#Zr_rare
            elif save_path_name == "Zr_mix":
                selected_element_list = ["Ca","Sr","Ti","Zr","Hf","Nb","Ta","Mo","W","Re","V","Cr","La","Pr"]#Zr_mix
            elif save_path_name == "Cu_total":
                selected_element_list = ["Mg","Cr","Mn","Fe","Co","Ni","Cu","Ag","Au","Zn","Ga","Ge","Sn","Sb","Bi"]#Cu_total
            elif save_path_name == "Cu_single":
                selected_element_list = ["Mg","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","Sn","Sb","Bi"]#Cu_single
            elif save_path_name == "Cu_mix":
                selected_element_list = ["Mg","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","Sn","Sb","Bi","Li","Pd","Al","La"]#Cu_mix
            elif save_path_name == "Mg_total":
                selected_element_list = ["Mg","Cr","Fe","Co","Ni","Cu","Ag","Zn","Ga","Ge","Sn","Sb","Bi"]#Mg_total
            elif save_path_name == "Mg_single":
                selected_element_list = ["Mg","Cr","Fe","Co","Ni","Cu","Zn","Ga","Ge","Sn","Sb","Bi"]#Mg_single
            elif save_path_name == "Mg_mix":
                selected_element_list = ["Mg","Cr","Fe","Co","Ni","Cu","Ag","Zn","Ga","Ge","Sn","Sb","Bi","Li","V","Mn","La","Al"]#Mg_mix
            elif save_path_name == "Ti_total":
                selected_element_list = ["Li","Ca","Sc","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Ru","Os","Co","Rh","Pd","Ir","Pt","Ag","Au","La"]#Ti_total
            elif save_path_name == "Ti_single":
                selected_element_list = ["Li","Ca","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Co"]#Ti_single
            elif save_path_name == "Ti_noble":
                selected_element_list = ["Li","Ca","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Ru","Os","Co","Rh","Pd","Ir","Pt","Ag","Au"]#Ti_noble
            elif save_path_name == "Ti_rare":
                selected_element_list = ["Li","Ca","Sc","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Co","La"]#Ti_rare
            elif save_path_name == "Ti_mix" or save_path_name == "Ti_mix_2":
                selected_element_list = ["Li","Ca","Ti","V","Nb","Ta","Cr","Mo","W","Mn","Re","Fe","Co","Mg","Zr","Hf","Ni","Cu","Sn","Bi","Nd","Pm"]#Ti_mix
            
            all_combinations = list(itertools.combinations(selected_element_list, element_num))
            all_combinations = [" ".join(list(ele)) for ele in all_combinations]
            
            #selected_framework = ["fcc","bcc","hcp"]
            selected_framework = ["bcc"]
            
            hv_list = []
            main()
            painting_hv(hv_list)
        except:
            print("something wrong……")
            time.sleep(300)
            continue