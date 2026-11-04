# -*- coding: utf-8 -*-
"""
Created on Tue May 16 02:10:22 2023

@author: Realunknown
"""

import torch
import math
import torch.nn as nn
from sklearn.metrics import r2_score
from torch_geometric.nn import CGConv
from torch_geometric.nn import global_mean_pool as gmp
from torch_geometric.loader import DataLoader
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        
def normalization_pre(Dataset,classification):
    if classification == True:
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        sample_data_list = [Data_set[i].y for i in sample(range(len(Data_set)), 500)]
        sample_target = torch.stack(sample_data_list, dim=0)
        normalizer = Normalizer(sample_target)
    return normalizer

def data_input(data_path ,classification):
    Data_set = torch.load(os.path.join(data_path + 'cgcnn-voronoi-Formation_energy-metal-2.3w.pt'))
    new_data_set = []
    for data in Data_set:
        if classification == False:
            #if abs(data.y) < 400:
            if data.y < 0:
                 #data.y = torch.FloatTensor([np.log(data.y)])#log
                 data.y = torch.FloatTensor([data.y])#log
                 new_data_set.append(data)
            #else:
            #    continue
        else:
            data.y = torch.LongTensor([[int(data.y[0][0])]+[int(data.y[0][1])]])
    #random.shuffle(Data_set)
    random.shuffle(new_data_set)
    return new_data_set
#    Data_set = Data_set[0:20000]#Data volume.
    #return Data_set

def normalization(Data_set,normalizer):
    for data in Data_set:
        data.y = normalizer.norm(data.y)
    return Data_set

def adjust_learning_rate(optimizer, epoch, k, lr_raw):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = lr_raw * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(Data_set,normal,classification):
    model.train()
    train_acc = 0
    pred_acc = []
    y_acc = []
    loss_list = []
    for batch in train_loader:
        batch.to(device)
        pred = model(batch.x.long(), batch.edge_index.long(),batch.edge_attr.float(), batch.batch)
        if classification == True:
            loss = loss_fn(pred,batch.y.long())
        else:
            loss = loss_fn(pred,torch.unsqueeze(batch.y, dim=1))
#            print(torch.unsqueeze(batch.y, dim=1))
#            print(pred)
        if classification == True:
            for i in range(len(pred)):
                if np.exp(float(pred[i][0])) < 0.5:
                    pred_acc.append(1)
                else:
                    pred_acc.append(0)
            for i in range(len(batch.y)):
                if float(batch.y[i]) > 0.5:
                    y_acc.append(1)
                else:
                    y_acc.append(0)
            train_correct = 0
            for i in range(len(pred_acc)):
                if pred_acc[i] == y_acc[i]:
                    train_correct += 1
            train_acc += train_correct
        else:
            acc = 0
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred_acc = []
        y_acc = []
    mae_train = loss_mae(pred,torch.unsqueeze(batch.y, dim=1))*normal[1]
    acc = train_acc / (int(len(Data_set)*0.8))
    return loss, acc, mae_train

def test(Data_set,highest_acc,lowest_loss, classification, y_predicted,normal):
    model.eval()
    test_acc = 0
    pred_test_acc = []
    y_test_acc = []
    preds = []
    for batch in test_loader:
        batch.to(device)
        pred = model(batch.x.long(), batch.edge_index.long(),batch.edge_attr.float(),batch.batch)
        if classification == True:
            loss = loss_fn(pred,batch.y.long())
        else:
            loss = loss_fn(pred,torch.unsqueeze(batch.y, dim=1))
        if classification == True:
            for i in range(len(pred)):
                if np.exp(float(pred[i][0])) < 0.5:
                    pred_test_acc.append(1)
                else:
                    pred_test_acc.append(0)
            for i in range(len(batch.y)):
                if float(batch.y[i]) > 0.5:
                    y_test_acc.append(1)
                else:
                    y_test_acc.append(0)
            test_correct = 0
            for i in range(len(pred_test_acc)):
                if pred_test_acc[i] == y_test_acc[i]:
                    test_correct += 1
            test_acc += test_correct
            pred_test_acc = []
            y_test_acc = []
        else:
            for i in range(len(pred)):
                preds.append(float(pred[i]*normal[1] + normal[0]))
    acc = test_acc / (int(len(Data_set)*0.2))
    mae_val = loss_mae(pred,torch.unsqueeze(batch.y, dim=1))*normal[1]
    if classification == True:
        if acc > highest_acc:
            highest_acc = acc
            #torch.save(model.state_dict(),"./data/cgcnn_poisson_ratio_1w_231216_1.pth")
    else:
        if loss < lowest_loss and len(preds) != 0:
            lowest_loss = loss
            y_predicted = preds
            torch.save(model.state_dict(),"./energy/result/0606/pre_trained_model_normalized_cgcnn_formation_energy_metal_2.3w_240604.pth")#name
    return loss, acc ,highest_acc ,lowest_loss,mae_val, y_predicted

def get_train_val_test_loader(Data_set,normal,batch_size):
    random.shuffle(Data_set) 
    x_observed = []
    train_loader = DataLoader(Data_set[:int(len(Data_set)*0.8)], batch_size=batch_size, shuffle=False)
    test_save = Data_set[int(len(Data_set)*0.8):]
    test_loader = DataLoader(Data_set[int(len(Data_set)*0.8):], batch_size=batch_size,shuffle= False)
    for batch in test_loader:
        for i in range(len(batch.y)):
            x_observed.append(float(batch.y[i]*normal[1] + normal[0]))
    return train_loader, test_loader, x_observed,test_save

def draw_loss(losses,losses2,y_name):
    plt.figure(1,dpi = 300)
    Min_1 = min(losses)
    Min_2 = min(losses2)
    y_max = max(Min_1,Min_2)*6
    x1 = range(0,len(losses))
    x2 = range(0,len(losses2))
    p1 = plt.scatter(x1 ,losses,color='b',label='training',s=5)
    p2 = plt.scatter(x2 ,losses2,color='r',label='testing',s=5)
    plt.plot(x1,losses,linestyle='solid',color='blue',linewidth = 0.1)
    plt.plot(x2,losses2,linestyle='solid',color='red',linewidth = 0.1)
    plt.xticks(np.arange(0, len(losses), int(len(losses)/10)))
    plt.yticks(np.arange(0, y_max, float('%.1g'%(y_max/10))))
    plt.xlabel("Epoch", fontdict={'size': 16})
    plt.ylabel(y_name, fontdict={'size': 16})
    plt.ylim(0,y_max)
    plt.legend(loc='best')
    plt.grid(axis = 'both',linewidth = 0.3)
    plt.savefig(save_path + 'trianing_curve'+ y_name + '.png', dpi=600)
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
    plt.xlabel('x_observed/eV')
    plt.ylabel('y_pred/eV')
    x = range(Min,Max + 1)
    y = range(Min,Max + 1)
    plt.xlim(Min,Max)
    plt.ylim(Min,Max)
    plt.title("cgcnn: Formation_Energy")#name
    text_x = Min + (Max-Min)*0.1 if Max != 1 else -0.95
    text_y = Max - (Max-Min)*0.2 if Max != 1 else 0.55
    plt.text(text_x,text_y,'r^2 = {:.2f}'.format(r_squared)  + '\nr = {:.2f}'.format(np.sqrt(r_squared)) ,fontdict={"fontsize":10,"color":"k"})
    plt.plot(x,y,color = 'darkorange')
    plt.scatter(x_observed, y_predicted, s=10, marker='o')
    plt.savefig(save_path + 'r_square_curve.png', dpi=600)
    plt.show()
    return r_squared

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
        self.embedding_edge = nn.Linear( 41, edge_fea_len,bias = False)
        self.convs = nn.ModuleList([CGConv([atom_fea_len , atom_fea_len], dim = edge_fea_len, batch_norm = True)
                                    for _ in range(n_conv)])
        self.softpluses1 = nn.ModuleList([nn.Softplus()
                                     for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses2 = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            #self.dropout = nn.Dropout(p=0)
            self.dropout = nn.Dropout(p=0.2)
    
    def forward(self ,x ,edge_index ,edge_attr ,batch_index ):
        r"""
        """
        atom_fea = self.embedding(x)
        atom_fea = torch.reshape(atom_fea, (len(atom_fea), 200))
        edge_attr = self.embedding_edge(edge_attr)
        for conv_func, softplus in zip(self.convs, self.softpluses1):
                atom_fea = softplus(conv_func(atom_fea, edge_index , edge_attr))
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
    
if __name__ == '__main__' :
    data_path = '../data_set/energy/'#class
    save_path = "./energy/result/0606/"
    classification = False
    y_predicted = [0,0,0]
    Data_set = data_input(data_path,classification)
    for h in range(len(Data_set)):
        if len(Data_set[h].edge_index[0]) != len(Data_set[h].edge_attr):
            del Data_set[h]
            break
    normalizer = normalization_pre(Data_set,classification)
    Data_set = normalization(Data_set,normalizer)
    model = CGCNN_pyg_net(classification)
    print(model)
    if torch.cuda.device_count() > 1:
        #If there is more than one GPU device, use two GPUs simultaneously
        model = nn.DataParallel(model, [0, 1])
    model=model.to(device)
    
    if classification == True:
        loss_fn = torch.nn.NLLLoss()
    else:
        loss_fn = torch.nn.MSELoss()
    loss_mae = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay= 1e-6)
    normal = []
    #
    mean = normalizer.mean
    std = normalizer.std
    normal.append(mean)
    normal.append(std)
    print('training mean :{0[0]}, training std:{0[1]}'.format(normal))
    train_loader, test_loader, x_observed, test_save = get_train_val_test_loader(Data_set,normal,batch_size =2048)
    torch.save(test_save,save_path + 'test_datasets_formation_energy_230604.pt')#name
    torch.save(normal, save_path + 'normalization_formation_energy_230604.pt')
    
    scheduler = MultiStepLR(optimizer, milestones= [45,160,400], gamma=0.1)
    
    print("Start training ...")
    losses = []
    losses2 = []
    accs = []
    accs2 = []
    mae_train_list = []
    mae_val_list = []
    highest_acc = 0.70
    lowest_loss = 0.05
    for epoch in range(1501):
        loss,acc, mae_train= train(Data_set,normal,classification)
        scheduler.step()
        losses.append(float(loss))
        accs.append(float(acc))
        loss2, acc2, highest_acc,lowest_loss,mae_val,y_predicted = test(Data_set,highest_acc,lowest_loss,classification,y_predicted,normal)
        mae_train_list.append(float(mae_train))
        mae_val_list.append(float(mae_val))
        accs2.append(float(acc2))
        losses2.append(float(loss2))
        if epoch%5 == 0:
            if classification == True:
                print(f"Epoch {epoch} | Training loss {loss} | Training acc {acc}")
                print(f"Epoch {epoch} | val loss {loss2} | val acc {acc2}")
                print("**********************************************************")
            else:
                print(f"Epoch {epoch} | Training loss {loss} ")
                print(f"Epoch {epoch} | val loss {loss2} | val mae {mae_val}")
                print("**********************************************************")
    torch.save(losses,save_path + "loss_train_formation_energy_230604.pt")#name
    torch.save(losses2,save_path + "loss_val_formation_energy_230604.pt")
    #torch.save(accs,save_path + "/0404/acc_train_poisson_ratio.pt")
    #torch.save(accs2,save_path + "/0404/acc_val_poisson_ratio.pt")
    torch.save(x_observed,save_path +"x_observed_formation_energy_230604.pt")
    torch.save(y_predicted,save_path +"y_predicted_formation_energy_230604.pt")
    torch.save(mae_train_list,save_path + 'mae_train_list_formation_energy_230604.pt')
    torch.save(mae_val_list,save_path + 'mae_val_list_formation_energy_230604.pt')
    draw_loss(losses,losses2,"Loss")
    draw_loss(mae_train_list,mae_val_list,"Mae")
    r_square = draw_r_square(x_observed,y_predicted)