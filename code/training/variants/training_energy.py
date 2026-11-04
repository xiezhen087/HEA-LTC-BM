# -*- coding: utf-8 -*-
"""
Created on Tue May 16 02:10:22 2023

@author: Realunknown
"""

import torch
import torch.nn as nn
from torch_geometric.nn import CGConv
from torch_geometric.nn import global_mean_pool as gmp
from torch_geometric.loader import DataLoader
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from torch.optim.lr_scheduler import MultiStepLR

import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

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
        
def normalization_pre(Data_set,classification):
    if classification == True:
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        sample_data_list = [Data_set[i].y for i in sample(range(len(Data_set)), 500)]
        sample_target = torch.stack(sample_data_list, dim=0)
        normalizer = Normalizer(sample_target)
    return normalizer

def data_input(data_path ,classification):
    Data_set = torch.load(os.path.join(data_path + 'cgcnn-voronoi-Formation_energy.pt'))
    new_data_set = []
    for data in Data_set:
        if classification == False:
            if data.y < 0:
                #data.y = torch.FloatTensor([np.log(data.y + 1)])#log
                data.y = torch.FloatTensor([data.y])
                new_data_set.append(data)
            else:
                continue
        else:
            data.y = torch.LongTensor([[int(data.y[0][0])]+[int(data.y[0][1])]])
    random.shuffle(new_data_set)
#    Data_set = Data_set[0:20000]#Data volume.
    return new_data_set

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

def train(Data_set,classification,optimizer,loss_fn,model):
    model.train()
    train_acc = 0
    pred_acc = []
    y_acc = []
    for batch in train_loader:
        batch.to(device)
        pred = model(batch.x.long(), batch.edge_index.long(),batch.edge_attr.float(), batch.batch)
        if classification == True:
            loss = loss_fn(pred,batch.y.long())
        else:
            loss = loss_fn(pred,torch.unsqueeze(batch.y, dim=1))
#            print(torch.unsqueeze(batch.y, dim=1))
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
    acc = train_acc / (int(len(Data_set)*0.8))
    mae_train = loss_mae(pred,torch.unsqueeze(batch.y, dim=1))
    return loss, acc,mae_train

def test(Data_set,highest_acc,lowest_loss, classification,optimizer,loss_fn,model):
    model.eval()
    test_acc = 0
    pred_test_acc = []
    y_test_acc = []
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
    acc = test_acc / (int(len(Data_set)*0.2))
    mae_val = loss_mae(pred,torch.unsqueeze(batch.y, dim=1))
    if classification == True:
        if acc > highest_acc:
            highest_acc = acc
            torch.save(model.state_dict(),"./energy/result/0604/model_best_cgcnn_12w_formation_energy_for_hea_0604.pth")
    else:
        if loss < lowest_loss:
            lowest_loss = loss
            torch.save(model.state_dict(),"./energy/result/0604/model_best_cgcnn_12w_formation_energy_for_hea_0604.pth")
    return loss, acc ,highest_acc ,lowest_loss,mae_val

def get_train_val_test_loader(Data_set,batch_size):
    random.shuffle(Data_set) 
    train_loader = DataLoader(Data_set[:int(len(Data_set)*0.8)], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Data_set[int(len(Data_set)*0.8):], batch_size=batch_size,shuffle= False)
    return train_loader, test_loader

def draw_loss(losses,losses2):
    plt.figure(1)
    x1 = range(0,len(losses))
    x2 = range(0,len(losses2))
    p1 = plt.scatter(x1 ,losses,color='b',label='training',s=5)
    p2 = plt.scatter(x2 ,losses2,color='r',label='testing',s=5)
    plt.plot(x1,losses,linestyle='solid',color='blue',linewidth = 0.1)
    plt.plot(x2,losses2,linestyle='solid',color='red',linewidth = 0.1)
    plt.xticks(np.arange(0, len(losses), 100))
    plt.yticks(np.arange(0, 1e9, 1e8))
    plt.xlabel("Epoch", fontdict={'size': 16})
    plt.ylabel("loss", fontdict={'size': 16})
    plt.legend(loc='best')
    plt.grid(b=True,axis = 'both',linewidth = 0.3)
    plt.show()
    return 0
    
def objective(trial):
    # Search the learning rate
    lr = trial.suggest_categorical('lr', [1e-10, 1e-9, 1e-8, 1e-7 ,1e-6,1e-5])
    # Search the weight decay factor
    weight_decay = trial.suggest_categorical('weight_decay', [1e-6, 1e-4, 1e-2, 1e0])
    # Search the network architecture hyperparameters
    num_embeding_characters = trial.suggest_categorical('num_characters', [10, 20, 50, 100 , 200])
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 5)
    num_linear_layers = trial.suggest_int('num_linear_layers',1,5)
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256 , 512])
    dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.2])
    
    class CGCNN_pyg_net(torch.nn.Module):
        r"""
        Using GCNConv Layer in Pyg Framework to Reproduce CGCNN Model.
        `"Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties"
        <https://doi.org/10.1103/PhysRevLett.120.145301>`_
        paper
        
        ----------
        Parameters:
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
                embbeding_fea_len = int(num_embeding_characters/10),
                atom_fea_len = num_embeding_characters,
                orig_edge_fea_len = 41,
                edge_fea_len =16,
                h_fea_len=hidden_size,
                n_conv = num_conv_layers,
                n_h=num_linear_layers,
                classification = False):
            super(CGCNN_pyg_net,self).__init__()
            torch.cuda.manual_seed(123)
            
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
                self.dropout = nn.Dropout(p=dropout)
        
        def forward(self ,x ,edge_index ,edge_attr ,batch_index):
            r"""
            """
            atom_fea = self.embedding(x)
            atom_fea = torch.reshape(atom_fea, (len(atom_fea), num_embeding_characters))
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
            #print(out)
            if self.classification:
                out = self.logsoftmax(out)
            return out

    # Define the model
    model = CGCNN_pyg_net()
    #print(model)
    if torch.cuda.device_count() > 1:
        #If there is more than one GPU device, use two GPUs simultaneously
        model = nn.DataParallel(model, [0, 1])
    model=model.to(device)
    
    # Define the loss function and optimizer
    if classification == True:
        loss_fn = torch.nn.NLLLoss()
    else:
        loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Train the model
    min_mae = float('inf')
    mae_list = []
    #print("Start training ...")
    losses = []
    losses2 = []
    train_mae = []
    val_mae = []
    accs = []
    accs2 = []
    highest_acc = 0.70
    lowest_loss = 0.5
    for epoch in range(1000):
        loss,acc,mae_train = train(Data_set,classification,optimizer,loss_fn,model)
        losses.append(float(loss))
        accs.append(float(acc))
        train_mae.append(mae_train)
        # Evaluate model performance on the test set
        with torch.no_grad():
            loss2, acc2, highest_acc,lowest_loss,mae_val = test(Data_set,highest_acc,lowest_loss,classification,optimizer,loss_fn,model)
            accs2.append(float(acc2))
            losses2.append(float(loss2))
            val_mae.append(mae_val)
            #print(mae_val)
        trial.report(mae_val, epoch)
        trial_num = trial.number
        if (mae_val < min_mae) and all( float(mae_val) <= _ for _ in mae_main_loop_list) and ((epoch > 200 and trial_num > 0) or (trial_num == 0 and epoch > 500)):
            torch.save(model.state_dict(), "./energy/result/0604/model_bo_temp-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
        min_mae = min(min_mae,mae_val)
        mae_list.append(mae_val)
        if len(mae_list) > 20:
            mae_list.pop(0)
        min_mae_over_last_20_epochs = min(mae_list)
        if min_mae < min_mae_over_last_20_epochs:
            if all( float(min_mae) <= _ for _ in mae_main_loop_list) and epoch > 200:
                print( '  epoch: {}'.format(epoch))
                torch.save(losses, "./energy/result/0604/train_loss_temp-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
                torch.save(losses2, "./energy/result/0604/val_loss_temp-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
                torch.save(train_mae, "./energy/result/0604/train_mae_temp-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
                torch.save(val_mae, "./energy/result/0604/val_mae_temp-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
                mae_main_loop_list.append(float(min_mae))
                print('  Best_mae: {}'.format(float(min_mae)))
                parameter = []
                for key, value in trial.params.items():
                    para = []
                    para.append(key)
                    para.append(value)
                    parameter.append(para)
                torch.save(parameter, "./energy/result/0604/parameter-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
            #print(mae_main_loop_list)
            #print(min_mae)
            #print(epoch)
            break
    #draw_loss(losses,losses2)
    #return float(mae_val)
    return float(min_mae)
    
if __name__ == '__main__' :
    data_path = '../data_set/energy/'
    classification = False
    Data_set = data_input(data_path,classification)
    loss_mae = nn.L1Loss()
    for h in range(len(Data_set)):
        if len(Data_set[h].edge_index[0]) != len(Data_set[h].edge_attr):
            del Data_set[h]
            break
    normalizer = normalization_pre(Data_set,classification)
    normal = []
    mean = normalizer.mean
    std = normalizer.std
    normal.append(mean)
    normal.append(std)
    print('training mean :{0[0]}, training std:{0[1]}'.format(normal))
    Data_set = normalization(Data_set,normalizer)
    train_loader, test_loader = get_train_val_test_loader(Data_set,batch_size = 2048)
    torch.save(test_loader,"./energy/result/0604/test_loader_formation_energy_0604_temp.pt")
    torch.save(normalizer,"./energy/result/0604/normalizer_formation_energy_0604_temp.pt")

    
    mae_main_loop_list = [float('inf')]
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials= 500)
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))