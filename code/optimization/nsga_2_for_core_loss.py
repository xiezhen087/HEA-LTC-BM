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
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import pathlib
import random
import xgboost as xgb
from pymoo.indicators.hv import HV
from torch_geometric.loader import DataLoader
import sys
import torch
from pymatgen.core.periodic_table import Element
import itertools
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure, Lattice
import torch.nn as nn

# --- local modules: models/ and data_processing/ sit next to optimization/ ---
_CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _CODE_DIR.parent
for _p in (_CODE_DIR / "models", _CODE_DIR / "data_processing"):
    sys.path.insert(0, str(_p))
from mixmodel import MixModel
from data_combine import output_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== run configuration =====================
# All values below can be overridden through the command-line wrapper
# code/optimize.py (recommended), or edited here directly.
SELECTED_ELEMENTS = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Nb', 'Mo', 'Al', 'Y']
ELEMENT_NUM = 5                # number of components in the alloy
SELECTED_FRAMEWORK = ["fcc"]   # any subset of ["fcc", "bcc", "hcp"]
RATIO_MIN = 0.05               # lower bound of each atomic fraction
RATIO_MAX = None               # upper bound; None -> 1 / ELEMENT_NUM
GENERATIONS = 500              # number of NSGA-II generations
POPNUM = 100                   # population size
ETA = 1                        # mutation distribution parameter (Deb recommends 1)
OUT_DIR = "./opt_run"          # output directory for Pareto logs and figures
HEADLESS = True                # True: save figures to OUT_DIR; False: show interactively
TEMPLATE_ROOT = str(_REPO_ROOT / "initial_structure")
MODEL_DIR_BM = str(_REPO_ROOT / "pre_trained_models" / "bulk_modulus" / "result"
                   / "0625_loop_save_time_1_r_square_0.8799")
MODEL_DIR_LTC = str(_REPO_ROOT / "pre_trained_models" / "thermal_conduct" / "result"
                    / "0802_loop_save_time_0_r_square_0.872")
MODEL_FILE_BM = "model_mix_best_bulk_modulus_0625_except_pure_metal_loop_save_time_1_simple.pth"
MODEL_FILE_LTC = "model_mix_best_thermal_conduct_0802_except_pure_metal_loop_save_time_0.pth"
# =============================================================

# search-space tables, rebuilt by configure_run() / main()
all_combinations = list(itertools.combinations(SELECTED_ELEMENTS, ELEMENT_NUM))
all_combinations = [" ".join(list(ele)) for ele in all_combinations]
element_num = ELEMENT_NUM
selected_framework = SELECTED_FRAMEWORK
hv_list = []

# pre-trained models are loaded once and cached (identical weights, so all
# evaluations share the same model instance)
_MODEL_CACHE = {}


def _get_model(prop):
    """Lazy-load and cache the Mix-GNN model for 'bulk_modulus' or 'thermal_conduct'."""
    if prop in _MODEL_CACHE:
        return _MODEL_CACHE[prop]
    if prop == "bulk_modulus":
        model_dir, model_file = MODEL_DIR_BM, MODEL_FILE_BM
    else:
        model_dir, model_file = MODEL_DIR_LTC, MODEL_FILE_LTC
    model = MixModel()
    if torch.cuda.device_count() > 1:
        # if more than one GPU device is available, use both GPUs at the same time
        model = nn.DataParallel(model, [0, 1])
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, model_file),
                                     map_location=torch.device('cpu')))
    normalization = torch.load(os.path.join(model_dir, "normalization.pt"))
    model.eval()
    _MODEL_CACHE[prop] = (model, normalization)
    return _MODEL_CACHE[prop]


def configure_run(elements=None, n_components=None, frameworks=None,
                  ratio_min=None, ratio_max=None):
    """Rebuild the search-space tables from (possibly new) parameters."""
    global SELECTED_ELEMENTS, ELEMENT_NUM, RATIO_MIN, RATIO_MAX
    global SELECTED_FRAMEWORK, all_combinations, element_num, selected_framework
    if elements is not None:
        SELECTED_ELEMENTS = list(elements)
    if n_components is not None:
        ELEMENT_NUM = int(n_components)
    if frameworks is not None:
        SELECTED_FRAMEWORK = list(frameworks)
    if ratio_min is not None:
        RATIO_MIN = float(ratio_min)
    if ratio_max is not None:
        RATIO_MAX = None if ratio_max in (None, "None", 0) else float(ratio_max)
    all_combinations = list(itertools.combinations(SELECTED_ELEMENTS, ELEMENT_NUM))
    all_combinations = [" ".join(list(ele)) for ele in all_combinations]
    element_num = ELEMENT_NUM
    selected_framework = SELECTED_FRAMEWORK
 
class Individual(object):
    def __init__(self):
        self.solution = None
        self.objective = defaultdict()
 
        self.n = 0
        self.rank = 0
        self.S = []
        self.distance = 0
 
    def bound_process(self, bound_min, bound_max):
        for i, item in enumerate(self.solution):
            if item > bound_max[i]:
                self.solution[i] = bound_max[i]
            elif item < bound_min[i]:
                self.solution[i] = bound_min[i]
 
    def calculate_objective(self, objective_fun):
        self.objective = objective_fun(self.solution)
 
    def __lt__(self, other):
        v1 = list(self.objective.values())
        v2 = list(other.objective.values())
        for i in range(len(v1)):
            if v1[i] > v2[i]:# define the optimization direction (max/min)
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
        lattice_constant[0] = 6.1366
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
    # pure-metal template per lattice type; the species are fully replaced by
    # re_construct_structure() and the lattice constants are recomputed by
    # modify_lattice_constant(), so the choice of template metal is immaterial
    template = {"fcc": "fcc/POSCAR_Cu_fcc",
                "bcc": "bcc/POSCAR_W_bcc",
                "hcp": "hcp/POSCAR_Zr_hcp"}[framework]
    poscar = Poscar.from_file(os.path.join(TEMPLATE_ROOT, template), 'r')
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

    x_element_composition = list(range(len(all_combinations)))  # element-combination index range
    ratio_max = RATIO_MAX if RATIO_MAX is not None else 1 / element_num
    x_element_ratio = [0]*(element_num - 1)
    for i in range(element_num - 1):
        x_element_ratio[i] = [RATIO_MIN, ratio_max]  # atomic-fraction range
    x_framework = list(range(len(selected_framework)))  # framework index range

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
    return data
 
def main(generations=None, popnum=None, eta=None, out_dir=None, headless=None):
    generations = GENERATIONS if generations is None else int(generations)  # number of generations
    popnum = POPNUM if popnum is None else int(popnum)  # population size
    eta = ETA if eta is None else float(eta)  # mutation distribution parameter; larger values make offspring more likely to approach their parents. Deb recommends setting it to 1
    out_dir = OUT_DIR if out_dir is None else out_dir
    headless = HEADLESS if headless is None else bool(headless)
    os.makedirs(out_dir, exist_ok=True)

    # Problem definition
    ratio_max = RATIO_MAX if RATIO_MAX is not None else 1 / element_num
    poplength = element_num + 1  # dimension of a single individual solution vector
    bound_min = [0] + [RATIO_MIN] * (element_num - 1) + [0]  # domain
    bound_max = [len(all_combinations) - 1] + [ratio_max] * (element_num - 1) + [len(selected_framework) - 1]
    objective_fun = KUR  #  objective function
 
    # Generate the first-generation population
    P = [] # create an empty list P to store individual objects
    print("Generate a population...")
    for i in range(popnum):
        print("Generate individual index:{}".format(i))
        # generate popnum individuals in a loop and add them to list P.
        P.append(Individual())
        P[i].solution = Generate_individual()  # randomly generate a feasible solution for the individual
        P[i].calculate_objective(objective_fun)  # compute the objective function values
    fast_non_dominated_sort(P)
    
    Q = make_new_pop(P, eta, bound_min, bound_max, objective_fun)
 
    P_t = P
    Q_t = Q
 
    for gen_cur in range(generations):
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
        Q_n = make_new_pop(P_n, eta, bound_min, bound_max,
                           objective_fun) 
 

        P_t = P_n
        Q_t = Q_n
        plot_P(P_t, gen_cur + 1, out_dir=out_dir, headless=headless)
        if not headless:
            plt.pause(0.1)

    # save the final Pareto-front individuals and the hypervolume history
    torch.save(P_t, os.path.join(out_dir, "nsga_pareto_front_individuals.pt"))
    torch.save(hv_list, os.path.join(out_dir, "nsga_hypervolume_history.pt"))
    # plain-text export: one row per Pareto individual (no pickle needed).
    # NOTE: the two property columns are the de-normalized raw model outputs,
    # i.e. the same convention as the NSGA-II POSCAR filename values — BEFORE
    # the per-subspace linear calibration (TSPPES value mapping). They are
    # monotonic within the validated subspaces, so ranking is unaffected;
    # apply the subspace mapping to obtain physical units (GPa, W m-1 K-1).
    with open(os.path.join(out_dir, "pareto_front.csv"), "w", newline="") as fh:
        fh.write("elements,atomic_ratios,framework,"
                 "bulk_modulus_raw,lattice_thermal_conductivity_raw\n")
        for ind in P_t:
            elems = all_combinations[int(ind.solution[0])].replace(" ", "-")
            ratios = "-".join("{:.4f}".format(float(v))
                              for v in ind.solution[1:element_num]) \
                     + "-{:.4f}".format(1.0 - sum(float(v) for v in ind.solution[1:element_num]))
            fw = selected_framework[int(ind.solution[-1])]
            fh.write("{},{},{},{:.4f},{:.4f}\n".format(
                elems, ratios, fw, -ind.objective[1], -ind.objective[2]))
    print("Pareto front saved to:", os.path.abspath(out_dir))
    if not headless:
        plt.show()
 
    return 0
 
def fast_non_dominated_sort(P):
    F = defaultdict(list)
 
    for p in P:
        p.S = [] 
        p.n = 0 
        for q in P:
            if p < q:# dominance direction: choose whether to optimize toward the maximum or minimum objective values
                p.S.append(q)
            elif q < p:# dominance direction: choose whether to optimize toward the maximum or minimum objective values
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
 

def make_new_pop(P, eta, bound_min, bound_max, objective_fun):
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
 
        
        Two_offspring = crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun)
 
        
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
 

def crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun):
    poplength = len(parent1.solution)
 
    offspring1 = Individual()
    offspring2 = Individual()
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
    
    offspring1.calculate_objective(objective_fun)
    offspring2.calculate_objective(objective_fun)
 
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

def goal_func_bulk_modulus(data): #bulk_modulus
    model, normalization = _get_model("bulk_modulus")

    val_loader = DataLoader([data], batch_size= 1,shuffle= False)
    for batch in val_loader:
        for i in range(1,len(batch[0])):
            batch[0][i] = batch[0][i].to(device)
        batch[1].to(device)
        with torch.no_grad():
            energy_per_atom = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
    #print("Bulk Modulus:{}".format(energy_per_atom[0][0]))
    y = float(energy_per_atom[0][0])*float(normalization[1]) + float(normalization[0])
    #print("y:{}".format(y))
    return y

"""
def goal_func_2(test_x):
    y_real_x_2 = test_x[3]*test_x[8]
    return np.log(y_real_x_2)
"""
def goal_func_thermal_conduct(data):#lattice_thermal_conduct
    model, normalization = _get_model("thermal_conduct")

    val_loader = DataLoader([data], batch_size= 1,shuffle= False)
    for batch in val_loader:
        for i in range(1,len(batch[0])):
            batch[0][i] = batch[0][i].to(device)
        batch[1].to(device)
        with torch.no_grad():
            energy_per_atom = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
    #print("Thermal Conduct:{}".format(energy_per_atom[0][0]))
    y = float(energy_per_atom[0][0])*float(normalization[1]) + float(normalization[0])
    #print("y:{}".format(y))
    return y

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

def KUR(x_individual):
    data_tuple = individual_to_data(x_individual)
    
    f = defaultdict(float)
 
    f[1] = 0
    f[2] = 0
    
    f[1] = -goal_func_bulk_modulus(data_tuple[0])#max
    f[2] = -goal_func_thermal_conduct(data_tuple[1])#max
    
    return f
 
def plot_P(P, gen_cur, out_dir=None, headless=True):
    X = []
    Y = []
    for ind in P:
        X.append(-ind.objective[1])
        Y.append(-ind.objective[2])

    fig, ax = plt.subplots(figsize=(8,5),dpi = 600)
    ax.scatter(X, Y, s = 80, marker='o',edgecolors='darkgoldenrod',facecolors='gold', label='NSGA-2:Multi-Objective Optimization')
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.grid()
    plt.title("Pareto Front Plot Generation:{}".format(gen_cur))
    if headless:
        out_dir = OUT_DIR if out_dir is None else out_dir
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "pareto_front_gen_{}.png".format(gen_cur)))
        plt.close(fig)
    else:
        plt.show()
    hv = calculate_hv(X,Y)
    hv_list.append(hv)

def painting_hv(hv_list, out_dir=None, headless=True):
    x = np.linspace(0,len(hv_list),len(hv_list))
    plt.figure(dpi = 600)
    plt.plot(x, hv_list, color='darkgoldenrod')
    plt.grid()
    plt.title('Hypervolume with NSGA-II')
    if headless and hv_list:
        out_dir = OUT_DIR if out_dir is None else out_dir
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "hypervolume_history.png"))
        plt.close()
    else:
        plt.show()
    return 0

# Entering main function
if __name__ == '__main__':
    # standalone run uses the module-level configuration above;
    # prefer the CLI wrapper: python code/optimize.py --method nsga2 ...
    main()
    painting_hv(hv_list)