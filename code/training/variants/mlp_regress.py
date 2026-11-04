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

import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
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
cont_c = [1,4,5,6,7,12,13,14,16,17,18,20,21,22,23,24,25,26,27,28,29]

def change_type(data_mlp,labels):
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

def package(data,labels):
    data_fin = []
    for i in range(len(data)):
        if labels[i] > 1000 or labels[i] < 0:
            continue
        data_fin_line = []
        embedding_a_data = []
        embedding_b_data = []
        embedding_c_data = []
        embedding_d_data = []
        continuous_data = []
        for j in range(len(data[i])):
            if j in embedding_a:
                embedding_a_data.append(data[i][j])
            elif j in embedding_b:
                embedding_b_data.append(data[i][j])
            elif j in embedding_c:
                embedding_c_data.append(data[i][j])
            elif j in cont_c:
                continuous_data.append(data[i][j])
        data_fin_line.append(torch.LongTensor(embedding_a_data))
        data_fin_line.append(torch.LongTensor(embedding_b_data))
        data_fin_line.append(torch.LongTensor(embedding_c_data))
        data_fin_line.append(torch.FloatTensor(continuous_data))
        data_fin_line.append(torch.FloatTensor([np.log(labels[i] + 1)]))
        data_fin.append(data_fin_line)
    return data_fin

class mlp(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,p = 0.3,p1 = 0.3):#or 0.3, which gives less overfitting
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

        self.linearE = torch.nn.Linear(21,60)#Xie's embedding directly uses nn.Linear
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

def train(data,normal):
    model.train()
    train_mae_total = []
    for batch in train_loader:
        optimizer.zero_grad()

        #print(batch[2])
        pred = model(batch[0],batch[1],batch[2],batch[3])
        #print(pred)
        loss = loss_fn(pred, batch[4])
        for i in range(len(pred)):#compute mae
            mae_i = abs(pred[i]-batch[4][i])*normal[1]
            train_mae_total.append(mae_i)
            #print('pred:{}'.format(pred[i]))
            #print('label:{}'.format(batch[4][i]))
            #print('loss:{}'.format(loss))
        loss.backward()
        optimizer.step()
    mae = float(sum(train_mae_total)/len(train_mae_total))
    return loss, mae

def test(data,lowest_loss,normal,y_predicted):
    model.eval()
    val_mae_total = []
    preds = []
    for batch in test_loader:
        optimizer.zero_grad()

#        print(batch[2])
        pred = model(batch[0],batch[1],batch[2],batch[3])
#        print(pred)
        loss = loss_fn(pred, batch[4])
        for i in range(len(pred)):
            mae_i = abs(pred[i]-batch[4][i])*normal[1]
            val_mae_total.append(mae_i)
            preds.append(float(pred[i]*normal[1] + normal[0]))
    mae = float(sum(val_mae_total)/len(val_mae_total))
    if loss < lowest_loss and len(preds) != 0:
        lowest_loss = loss
        y_predicted = preds
        #torch.save(model.state_dict(), save_path + "model_mix_best_0412.pth")
    return loss, mae,lowest_loss,y_predicted

def draw_loss(losses,losses2,y_name):
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
    #plt.savefig(save_path + '/trianing_curve'+ y_name + '.png', dpi=600)
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
    plt.title("mlp: Phonon_thermal_capacity")#name
    text_x = Min + (Max-Min)*0.1 if Max != 1 else -0.95
    text_y = Max - (Max-Min)*0.2 if Max != 1 else 0.55
    plt.text(text_x,text_y,'r^2 = {:.2f}'.format(r_squared)  + '\nr = {:.2f}'.format(np.sqrt(r_squared)) ,fontdict={"fontsize":10,"color":"k"})
    plt.plot(x,y,color = 'darkorange')
    plt.scatter(x_observed, y_predicted, s=10, marker='o')
    #plt.savefig(save_path + '/r_square_curve.png', dpi=600)
    plt.show()
    return r_squared

if __name__ == '__main__' :
    input_path = "../data_set/thermal_conduct/"

    input_size, hidden_size, output_size = 7,128, 1
    model = mlp(input_size, hidden_size, output_size)
    print(model)
    
    data_raw = pd.read_csv(input_path + "features_normalized_thermal_conduct_log.csv")
    labels = data_raw['target']
    data_mlp = data_raw.drop('target', axis = 1).values.tolist()
    labels = labels.values.tolist()
    
    data = change_type(data_mlp, labels)
    data = package(data,labels)
    
    y = []
    for d in data:
        y.append(float(d[-1]))
    normal = []
    normal.append(sum(y)/len(y))
    normal.append(np.std(y))
    print('training mean :{0[0]}, training std:{0[1]}'.format(normal))
    
    for d in data:
        d[-1][0] = (d[-1][0]-normal[0])/normal[1]
    
    LR = 1e-3
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    x_observed = []
    data_size = len(data)
    batch_size = 2048
    random.shuffle(data)
    train_loader = DataLoader(data[:int(data_size*0.8)], batch_size=batch_size, shuffle=True)
    for d in data[int(data_size*0.8):]:
        x_observed.append(float(d[-1])*normal[1]+normal[0])
    test_loader = DataLoader(data[int(data_size*0.8):], batch_size=batch_size, shuffle = False)
    

    print("Start training ...")
    losses_train = []
    losses_val = []
    maes_train = []
    maes_val = []
    y_predicted = []
    lowest_loss = 0.50
    i = 0
    for epoch in range(301):
          loss_train, mae_train = train(data,normal)
          losses_train.append(float(loss_train))
          maes_train.append(float(mae_train))
          loss_val,mae_val, lowest_loss,y_predicted = test(data,lowest_loss,normal,y_predicted)
          losses_val.append(float(loss_val))
          maes_val.append(float(mae_val))
          if epoch%10 == 0:
              print(f"Epoch {epoch} | Training loss {loss_train} | Training acc {mae_train}")
              print(f"Epoch {epoch} | Testing loss {loss_val} | Testing acc {mae_val}")
              print("**********************************************************")
          #break
    draw_loss(losses_train,losses_val,"loss")
    draw_loss(maes_train,maes_val,"mae")
    r_square = draw_r_square(x_observed,y_predicted)