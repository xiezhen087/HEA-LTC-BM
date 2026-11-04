# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:54:00 2023

@author: Realunknown
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:24:09 2022

@author: Realunknown
"""

pure_metal_id = ["mp-134","mp-23","mp-30","mp-126","mp-2","mp-81","mp-117","mp-102","mp-13","mp-90","mp-129","mp-75","mp-91","mp-146","mp-72","mp-123","mp-131","mp-8566"]

import csv
import numpy as np
from torch_geometric.nn import CGConv
from torch_geometric.nn import global_mean_pool as gmp
from sklearn.preprocessing import StandardScaler
import os
import shutil
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.metrics import r2_score
import math

mp_id = [0]
type_a = [2,3,10,11]
type_b = [8,9]
type_c = [1,4,5,6,7,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
embedding_a = [2,9]
embedding_b = [8]
embedding_c = [10,11]
#cont_c = [1,4,5,6,7,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]#15
cont_c = [1,4,5,6,7,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# ===================== run configuration =====================
# Override through the command-line wrapper code/train.py (recommended),
# or edit here directly.
PROPERTY = "thermal_conduct"     # "thermal_conduct" or "bulk_modulus"
INPUT_PATH = "../data_set/thermal_conduct/"   # folder with the two inputs below
SAVE_ROOT = "./thermal_conduct/result/"
SAVE_TEMP = "./thermal_conduct/result/model_temp/"
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2048
TRAIN_RATIO = 0.8
MIN_EPOCHS = 3000        # keep training at least this many epochs
MAX_EPOCHS = 12000       # hard cap per outer loop
R2_TARGET = 0.88         # stop an outer loop once R^2 (rounded to 2 dp) reaches this
R2_SAVE = 0.87           # save the model once R^2 (rounded) reaches this
MAX_LOOPS = 2000         # outer-loop cap
MAX_MODELS = 5           # stop after saving this many qualified models
# =============================================================

# property-dependent input/output file names
_PROP_FILES = {
    "thermal_conduct": {
        "features_csv": "features_normalized_thermal_conduct_log.csv",
        "cgcnn_pt": "cgcnn-voronoi-thermal_conductivity-HEA.pt",
        "model_name": "thermal_conduct",
    },
    "bulk_modulus": {
        "features_csv": "features_normalized_bulk_modulus_log.csv",
        "cgcnn_pt": "cgcnn-voronoi-bulk_modulus-HEA.pt",
        "model_name": "bulk_modulus",
    },
}

def change_type_mlp(data_mlp,labels):
    data = []
    for i in range(len(data_mlp)):
        data_line = []
        for j in range(len(data_mlp[0])):
            if j in mp_id:
                data_line.append(data_mlp[i][j])
            if (j in type_a) or (j in type_b):
                data_mlp[i][j] = int(data_mlp[i][j])
                data_line.append(data_mlp[i][j])
            if j in type_c:
                data_line.append(data_mlp[i][j])
        data.append(data_line)
    return data

def package_mlp(data,labels):
    data_fin = []
    for i in range(len(data)):
        #if labels[i] > 1000 or labels[i] < 0:
        #    continue
        data_fin_line = []
        mp_id_data = []
        embedding_a_data = []
        embedding_b_data = []
        embedding_c_data = []
        embedding_d_data = []
        continuous_data = []
        for j in range(len(data[i])):
            if j in mp_id:
                mp_id_data.append(data[i][j])
            elif j in embedding_a:
                embedding_a_data.append(data[i][j])
            elif j in embedding_b:
                embedding_b_data.append(data[i][j])
            elif j in embedding_c:
                embedding_c_data.append(data[i][j])
            elif j in cont_c:
                continuous_data.append(data[i][j])
        data_fin_line.append(mp_id_data)
        data_fin_line.append(torch.LongTensor(embedding_a_data))
        data_fin_line.append(torch.LongTensor(embedding_b_data))
        data_fin_line.append(torch.LongTensor(embedding_c_data))
        data_fin_line.append(torch.FloatTensor(continuous_data))
        data_fin_line.append(torch.FloatTensor([np.log(labels[i] + 1)]))
        data_fin.append(data_fin_line)
    return data_fin

def cgcnn_data(input_path):
    data_set = torch.load(os.path.join(input_path, _PROP_FILES[PROPERTY]["cgcnn_pt"]))
    for i in range(len(data_set)):
        data_set[i].y = torch.FloatTensor([np.log(data_set[i].y + 1)])#log
        for j in range(len(data_set[i].x)):
            if (type(data_set[i].x[j]) is not int) and (type(data_set[i].x[j]) is not float):
                data_set[i].x[j] = torch.tensor(0.0)
    return data_set

def package_total_data(data_mlp,data_cgcnn):
   #Merge the GCN and MLP data parts
    data = []
    DATA = []
    for i in range(len(data_mlp)):
        if len(data_cgcnn[i].edge_attr) != len(data_cgcnn[i].edge_index[0]):#one sample is missing both edge features
            continue
        if data_mlp[i][0][0] in pure_metal_id:
            continue
        DATA.append(data_mlp[i])
        DATA.append(data_cgcnn[i])
        data.append(DATA)
        DATA = []
    random.shuffle(data)
    return data

class CGCNN_pyg_net(torch.nn.Module):
    r"""
    Using GCNConv Layer in Pyg Framework to Reproduce CGCNN Model.
    `"Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties"
    <https://doi.org/10.1103/PhysRevLett.120.145301>`_
    paper
    
    ----------
    Parameter:
    orig_atom_fea_len: 
        Number of atom features in the input.
    atom_fea_len: 
        Number of hidden atom features in the convolutional layers.
    h_fea_len:
        Number of hidden features after pooling.
    n_conv: 
        Number of convolutional layers.
    n_h:
        Number of hidden layers after pooling.
    classification: 
        What task to complete. 'Ture' for classification and 'False' for regression.
    
    """
    def __init__(
            self,
            orig_atom_fea_len = 10,
            continous_fea_len = 8,
            embbeding_fea_len = 20,
            atom_fea_len = 200,
            orig_edge_fea_len = 41,
            edge_fea_len =16,
            h_fea_len= 64,
            n_conv = 4,
            n_h= 2,
            classification = False):
        super(CGCNN_pyg_net,self).__init__()
        torch.manual_seed(123)
        
        self.classification = classification
        self.embedding = nn.Embedding(96,embbeding_fea_len)
        self.continous_linear = nn.Linear( 3, continous_fea_len,bias = True)
        self.embedding_edge = nn.Linear( 41, edge_fea_len,bias = False)
        self.convs = nn.ModuleList([CGConv([atom_fea_len + continous_fea_len , atom_fea_len + continous_fea_len], dim = edge_fea_len, batch_norm = True)
                                    for _ in range(n_conv)])
        self.softpluses1 = nn.ModuleList([nn.Softplus()
                                     for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len + continous_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses2 = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 80)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(p=0)
    
    def forward(self ,x ,edge_index ,edge_attr ,batch_index):
        r"""
        """
        atom_fea = self.embedding(x[:, :10].long())
        atom_fea = torch.reshape(atom_fea, (len(atom_fea), 200))
        continous_fea = self.continous_linear(x[:,-3:])
        fea_all = torch.cat((atom_fea,continous_fea),dim = 1)
        edge_attr = self.embedding_edge(edge_attr)
        for conv_func, softplus in zip(self.convs, self.softpluses1):
                atom_fea = softplus(conv_func(fea_all, edge_index , edge_attr))
        crys_fea = gmp(atom_fea, batch_index)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses2'):
            for fc, softplus in zip(self.fcs, self.softpluses2):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

class mlp(nn.Module):
    def __init__(self, input_size =7, hidden_size=128, output_size=80 ,p = 0.3,p1 = 0.3):#or 0.3, which gives less overfitting
        super().__init__()
        torch.manual_seed(42)
        self.embedding_a = torch.nn.Embedding(350, 8)
        self.flatten = torch.nn.Flatten(start_dim = 1,end_dim = -1)
        self.linearA = torch.nn.Linear(16,20)
        self.reluA = torch.nn.ReLU()
        self.dropoutA = nn.Dropout(p1)
#        self.BN1 = nn.BatchNorm1d(4)
#        self.BN2 = nn.BatchNorm1d(hidden_size)

        #self.embedding_b = torch.nn.Embedding(80, 8)
        #self.linearB = torch.nn.Linear(16,8)
        #self.reluB = torch.nn.ReLU()
        #self.dropoutB = nn.Dropout(p1)
        
        self.embedding_c = torch.nn.Embedding(10, 8)
        self.linearC = torch.nn.Linear(8,8)
        self.reluC = torch.nn.ReLU()
        self.dropoutC = nn.Dropout(p1)
        
        self.embedding_d = torch.nn.Embedding(2,8)
        self.linearD = torch.nn.Linear(16,8)
        self.reluD = torch.nn.ReLU()
        self.dropoutD = nn.Dropout(p1)        

        self.linearE = torch.nn.Linear(23,60)#Xie's embedding directly uses nn.Linear
        self.reluE = torch.nn.ReLU()
        self.dropoutE = nn.Dropout(p1)

        #self.linear1 = torch.nn.Linear(96, hidden_size)
        self.linear1 = torch.nn.Linear(96, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = nn.Dropout(p)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = nn.Dropout(p)
        self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = nn.Dropout(p)
        self.linear4 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu4 = torch.nn.ReLU()
        self.dropout4 = nn.Dropout(p)
        self.linear5 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu5 = torch.nn.ReLU()
        self.dropout5 = nn.Dropout(p)
        self.linear6 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu6 = torch.nn.ReLU()
        self.dropout6 = nn.Dropout(p)
        self.linear7 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu7 = torch.nn.ReLU()
        self.dropout7 = nn.Dropout(p)
        self.linear8 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu8 = torch.nn.ReLU()
#        self.sigmoid2 = torch.nn.Sigmoid()
        self.lineart1 = torch.nn.Linear(hidden_size,hidden_size)
        self.dropout1 = nn.Dropout(p = 0.6)
        self.relu9 = torch.nn.ReLU()
        self.out = torch.nn.Linear(hidden_size, output_size)
#        self.downsample = downsample

    def forward(self, inputA,inputC,inputD, inputE):
        A = self.embedding_a(inputA)
        A = self.flatten(A)
        A = self.linearA(A)
        A = self.reluA(A)
        out_A = self.dropoutA(A)

        #B = self.embedding_b(inputB)
        #B = self.flatten(B)
        #B = self.linearB(B)
        #B = self.reluB(B)
        #out_B = self.dropoutB(B)
        
        C = self.embedding_c(inputC)
        C = self.flatten(C)
        C = self.linearC(C)
        C = self.reluC(C)
        out_C = self.dropoutC(C)
        
        D = self.embedding_d(inputD)
        D = self.flatten(D)
        D = self.linearD(D)
        D = self.reluD(D)
        out_D = self.dropoutD(D)

        E = self.linearE(inputE)
        E = self.reluE(E)
        out_E = self.dropoutE(E)
        
        #x = torch.cat((out_A,out_B,out_C,out_D,out_E), dim=1)#Thermal conduct
        #x = torch.cat((out_A,out_C,out_D,out_E), dim=1)
        x = torch.cat((out_A,out_C,out_D,out_E), dim=1)
        #x = out_E
        x = self.linear1(x)
        hidden = self.relu1(x)
        hidden = self.dropout1(hidden)
        x_1 = hidden
        hidden = self.linear2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.dropout2(hidden)
        x_2 = hidden
        hidden = self.linear3(hidden)
        hidden = self.relu3(hidden)
        hidden = self.dropout3(hidden)
        x_3 = hidden
        hidden = self.linear4(hidden)
        hidden = self.relu4(hidden)
        hidden = self.dropout4(hidden)
        x_4 = hidden
        hidden = self.linear5(hidden)
        hidden = self.relu5(hidden)
        hidden = self.dropout5(hidden)
        x_5 = hidden
        hidden = self.linear6(hidden)
        hidden = self.relu6(hidden)
        hidden = self.dropout6(hidden)
        x_6 = hidden
        hidden = self.linear7(hidden)
        hidden = self.relu7(hidden)
        hidden = self.dropout7(hidden)
        x_7 = hidden
        hidden = self.linear8(hidden)
        hidden = self.relu8(hidden)
        hidden = hidden + x + x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7
        hidden = self.lineart1(hidden)
        hidden = self.dropout1(hidden)
        hidden = self.relu9(hidden)
        #hidden_all = self.tanh1(hidden_all)
        out = self.out(hidden)
        return out
    
class MixModel(nn.Module):
    def __init__(self,
                 mlp_hidden_size = 128, 
                 mlp_output_size = 80,
                 output_size = 1):
        super().__init__()
        self.mlp_model = mlp()
        self.gcn_model = CGCNN_pyg_net()
        self.lineart1 = torch.nn.Linear(80,128)
        self.dropout1 = nn.Dropout(p = 0.6)
        self.relu1 = torch.nn.ReLU()
        #self.tanh1 = torch.nn.Tanh()
        self.out = torch.nn.Linear(128, output_size)#total number of output parameters
        
    def forward(self,data_mlp,data_gcn_x,data_gcn_edge,data_gcn_edge_attr,batch_index):
        mlp_out = self.mlp_model(data_mlp[1],data_mlp[2],data_mlp[3],data_mlp[4])
        gcn_out = self.gcn_model(data_gcn_x, data_gcn_edge,data_gcn_edge_attr,batch_index)
             
        result_input = torch.mul(mlp_out , gcn_out)
        #result_input = gcn_out
        hidden_all = self.lineart1(result_input)
        hidden_all = self.dropout1(hidden_all)
        hidden_all = self.relu1(hidden_all)
        #hidden_all = self.tanh1(hidden_all)
        result_out = self.out(hidden_all)
        #out = torch.softmax(result_out,dim = 1)
        return result_out

def train(data,normal):
    model.train()
    train_mae_total = []
    for batch in train_loader:
        for i in range(len(batch[0])):
            if i > 0:
                batch[0][i] = batch[0][i].to(device)
        batch[1].to(device)

        #print(batch[2])
        pred = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
        #print(pred)
        loss = loss_fn(pred,torch.unsqueeze(batch[1].y, dim=1))
        for i in range(len(pred)):#compute mae
            mae_i = abs(pred[i]-batch[1].y[i])*normal[1]
            train_mae_total.append(float(mae_i))
            #print('pred:{}'.format(pred[i]))
            #print('label:{}'.format(batch[4][i]))
            #print('loss:{}'.format(loss))
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    mae = float(sum(train_mae_total)/len(train_mae_total))
    return loss, mae

def test(data,lowest_loss,normal,y_predicted):
    model.eval()
    val_mae_total = []
    preds = []
    for batch in test_loader:
        for i in range(len(batch[0])):
            if i > 0:
                batch[0][i] = batch[0][i].to(device)
        batch[1].to(device)
        optimizer.zero_grad()

#        print(batch[2])
        with torch.no_grad():
            pred = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
#        print(pred)
            loss = loss_fn(pred,torch.unsqueeze(batch[1].y, dim=1))
            for i in range(len(pred)):
                mae_i = abs(pred[i]-batch[1].y[i])*normal[1]
                val_mae_total.append(float(mae_i))
                preds.append(float(pred[i]*normal[1] + normal[0]))
            mae = float(sum(val_mae_total)/len(val_mae_total))
    if loss < lowest_loss and len(preds) != 0:
        lowest_loss = loss
        y_predicted = preds
        torch.save(model.state_dict(), save_path_temp + "model_mix_best_" + _PROP_FILES[PROPERTY]["model_name"] + "_loop_save_time_"+ str(save_time) + ".pth")
    return loss, mae,lowest_loss,y_predicted

def draw_loss(losses,losses2,y_name,save_path):
    plt.figure(1,dpi = 300)
    Min_1 = min(losses)
    Min_2 = min(losses2)
    y_max = max(Min_1,Min_2)*6
    x1 = range(0,len(losses))
    x2 = range(0,len(losses2))
    plt.scatter(x1 ,losses,color='b',label='training',s=5)
    plt.scatter(x2 ,losses2,color='r',label='testing',s=5)
    plt.plot(x1,losses,linestyle='solid',color='blue',linewidth = 0.1)
    plt.plot(x2,losses2,linestyle='solid',color='red',linewidth = 0.1)
    plt.xticks(np.arange(0, len(losses), int(len(losses)/10)))
    plt.yticks(np.arange(0, y_max, float('%.1g'%(y_max/10))))
    plt.xlabel("Epoch", fontdict={'size': 16})
    plt.ylabel(y_name, fontdict={'size': 16})
    plt.ylim(0,y_max)
    plt.legend(loc='best')
    plt.grid(axis = 'both',linewidth = 0.3)
    plt.savefig(save_path + '/trianing_curve'+ y_name + '.png', dpi=600)
    plt.show()
    return 0

def draw_r_square(x_observed,y_predicted):
    plt.figure(dpi = 300)
    r_squared = r2_score(x_observed, y_predicted)
    x_max = math.ceil(max(x_observed))
    x_min = math.floor(min(x_observed))
    y_max = math.ceil(max(y_predicted))
    y_min = math.floor(min(y_predicted))
    Max = max(x_max,y_max)
    Min = min(x_min,y_min)
    plt.xlabel('x_observed/log(J/K/mol)')
    plt.ylabel('y_pred/log(J/K/mol)')
    x = range(Min,Max + 1)
    y = range(Min,Max + 1)
    plt.xlim(Min,Max)
    plt.ylim(Min,Max)
    plt.title("mlp: Thermal_conduct")#name
    text_x = Min + (Max-Min)*0.1 if Max != 1 else -0.95
    text_y = Max - (Max-Min)*0.2 if Max != 1 else 0.55
    plt.text(text_x,text_y,'r^2 = {:.2f}'.format(r_squared)  + '\nr = {:.2f}'.format(np.sqrt(r_squared)) ,fontdict={"fontsize":10,"color":"k"})
    plt.plot(x,y,color = 'darkorange')
    plt.scatter(x_observed, y_predicted, s=10, marker='o')
    plt.savefig(save_path + '/r_square_curve.png', dpi=600)
    plt.show()
    return r_squared

def main():
    """Final training loop for the Mix-GNN models.

    Reads the module-level run configuration (PROPERTY, INPUT_PATH, SAVE_ROOT,
    LR, ...; set via code/train.py or edited directly above) and saves every
    model reaching R2_SAVE under SAVE_ROOT.
    """
    global train_loader, test_loader, model, optimizer, loss_fn
    global save_path_temp, save_time
    import datetime

    input_path = INPUT_PATH
    save_path_root = SAVE_ROOT
    save_path_temp = SAVE_TEMP
    os.makedirs(save_path_temp, exist_ok=True)

    features_csv = os.path.join(input_path, _PROP_FILES[PROPERTY]["features_csv"])
    cgcnn_pt = os.path.join(input_path, _PROP_FILES[PROPERTY]["cgcnn_pt"])
    for f in (features_csv, cgcnn_pt):
        if not os.path.isfile(f):
            raise FileNotFoundError(
                "Required training input not found: {}\n"
                "Build the dataset with the code/data_processing/ chain first, "
                "or download the dataset package (see README / Zenodo).".format(f))

    input_size, hidden_size, output_size = 7,128, 1

    data_raw = pd.read_csv(features_csv)
    labels = data_raw['target']
    data_mlp = data_raw.drop('target', axis = 1).values.tolist()
    labels = labels.values.tolist()
    
    data_mlp = change_type_mlp(data_mlp, labels)
    data_mlp = package_mlp(data_mlp,labels)
    
    data_cgcnn = cgcnn_data(input_path)
    data = package_total_data(data_mlp, data_cgcnn)
    
    data_size = len(data)
    batch_size = BATCH_SIZE
    
    y = []
    for d in data:
        y.append(float(d[1].y))
    normal = []
    normal.append(sum(y)/len(y))
    normal.append(np.std(y))
    print('training mean :{0[0]}, training std:{0[1]}'.format(normal))
    
    for d in data:
        d[1].y = (d[1].y-normal[0])/normal[1]
    
    print("Entering main loop...")
    loop_time = 0
    save_time = 0
    while(loop_time < MAX_LOOPS and save_time < MAX_MODELS):
        
        model = MixModel()
        if torch.cuda.device_count() > 1:
            #If there is more than one GPU device, use two GPUs simultaneously
            model = nn.DataParallel(model, [0, 1])
        model=model.to(device)
        print(model)
        
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay = WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        
        x_observed = []
        random.shuffle(data)
        train_loader = DataLoader(data[:int(data_size*TRAIN_RATIO)], batch_size=batch_size, shuffle=True)
        for d in data[int(data_size*TRAIN_RATIO):]:
            x_observed.append(float(d[1].y)*normal[1]+normal[0])
        test_loader = DataLoader(data[int(data_size*TRAIN_RATIO):], batch_size=batch_size, shuffle = False)

        print("Start training ...")
        losses_train = []
        losses_val = []
        maes_train = []
        maes_val = []
        y_predicted = []
        lowest_loss = 0.50
        i = 0
        epoch = 0
        r_squared = 0
        while(((epoch > MIN_EPOCHS) and (round(r_squared, 2) >= R2_TARGET)) == False and epoch < MAX_EPOCHS):
             loss_train, mae_train = train(data,normal)
             losses_train.append(float(loss_train))
             maes_train.append(float(mae_train))
             loss_val,mae_val, lowest_loss,y_predicted = test(data,lowest_loss,normal,y_predicted)
             losses_val.append(float(loss_val))
             maes_val.append(float(mae_val))
             #if epoch%10 == 0:
             #    print(f"Epoch {epoch} | Training loss {loss_train} | Training mae {mae_train}")
             #    print(f"Epoch {epoch} | Testing loss {loss_val} | Testing mae {mae_val}")
             #    print("**********************************************************")
             #break
             epoch += 1
             if len(y_predicted) != 0:
                 r_squared = r2_score(x_observed, y_predicted)
        print("save_time:{}".format(save_time))
        print("epoch:{}".format(epoch))
        print("r_squared:{}".format(r_squared))
        if round(r_squared, 2) >= R2_SAVE:
            tag = datetime.datetime.now().strftime("%m%d")
            save_path = save_path_root + tag + "_loop_save_time_" + str(save_time) + "_r_square_" + str(round(r_squared, 4)) + "/"
            os.makedirs(save_path_root, exist_ok=True)
            os.mkdir(save_path)
            torch.save(normal,save_path + "normalization.pt")
            torch.save(test_loader,save_path + "test_loader.pt")
            torch.save(losses_train, save_path + "train_loss_temp-" + str(0) + "-"+ str(float(mae_val))[:5]+".pt")
            torch.save(losses_val, save_path + "val_loss_temp-" + str(0) + "-"+ str(float(mae_val))[:5]+".pt")
            torch.save(mae_train, save_path + "train_mae_temp-" + str(0) + "-"+ str(float(mae_val))[:5]+".pt")
            torch.save(mae_val, save_path + "val_mae_temp-" + str(0) + "-"+ str(float(mae_val))[:5]+".pt")
            torch.save(x_observed, save_path + "x_observed-" + str(0) + "-"+ str(float(r_squared))[:5]+".pt")
            torch.save(y_predicted, save_path + "y_predicted-" + str(0) + "-"+ str(float(r_squared))[:5]+".pt")
            draw_loss(losses_train,losses_val,"loss",save_path)
            draw_loss(maes_train,maes_val,"mae",save_path)
            r_square = draw_r_square(x_observed,y_predicted)
            temp_name = "model_mix_best_" + _PROP_FILES[PROPERTY]["model_name"] + "_loop_save_time_" + str(save_time) + ".pth"
            shutil.copy(save_path_temp + temp_name, save_path + temp_name)
            del model
            save_time += 1
        loop_time += 1
    print("Done. {} model(s) saved under {}".format(save_time, os.path.abspath(save_path_root)))

if __name__ == '__main__':
    main()