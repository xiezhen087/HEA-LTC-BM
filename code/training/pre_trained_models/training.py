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

def data_input(data_path ,classification):
    Data_set = torch.load(os.path.join(data_path + 'mix-voronoi-bulk_modulus-except_pure_metal-HEA.pt'))
    new_data_set = []
    for data in Data_set:
        if classification == False:
            if data[1].y > 0:
                data[1].y = torch.FloatTensor([np.log(data[1].y)])#log
                #data.y = torch.FloatTensor([data.y])
                for j in range(len(data[1].x)):
                    if (type(data[1].x[j]) is not int) and (type(data[1].x[j]) is not float):
                        data[1].x[j] = torch.tensor(0.0)
                for j in range(len(data[0][-1])):
                    if (type(data[0][-1][j]) is not int) and (type(data[0][-1][j]) is not float):
                        data[0][-1][j] = torch.tensor(0.0)
                new_data_set.append(data)
            else:
                continue
        else:
            data.y = torch.LongTensor([[int(data.y[0][0])]+[int(data.y[0][1])]])
    random.shuffle(new_data_set)
#    Data_set = Data_set[0:20000]#Data volume.
    return new_data_set

def normalization(Data_set,normal):
    for data in Data_set:
        data[1].y = (data[1].y - normal[0])/normal[1]
    return Data_set

def adjust_learning_rate(optimizer, epoch, k, lr_raw):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = lr_raw * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(Data_set,optimizer,loss_fn,model):
    model.train()
    train_mae_total = []
    train_acc = 0
    pred_acc = []
    y_acc = []
    for batch in train_loader:
        for i in range(len(batch[0])):
            if i > 0:
                batch[0][i] = batch[0][i].to(device)
        batch[1].to(device)
        pred = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
        loss = loss_fn(pred,torch.unsqueeze(batch[1].y, dim=1))
        for i in range(len(pred)):
            mae_i = abs(pred[i]-batch[1].y[i])*normal[1]
            train_mae_total.append(mae_i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    mae_train = loss_mae(pred,torch.unsqueeze(batch[1].y, dim=1))
    return loss, mae_train

def test(Data_set,lowest_loss,optimizer,loss_fn,model):
    model.eval()
    preds = []
    val_mae_total = []
    for batch in test_loader:
        for i in range(len(batch[0])):
            if i > 0:
                batch[0][i] = batch[0][i].to(device)
        batch[1].to(device)
        with torch.no_grad():
            pred = model(batch[0],batch[1].x.float(), batch[1].edge_index.long(),batch[1].edge_attr.float(), batch[1].batch)
            loss = loss_fn(pred,torch.unsqueeze(batch[1].y, dim=1))
            for i in range(len(pred)):
                mae_i = abs(pred[i]-batch[1].y[i])*normal[1]
                val_mae_total.append(mae_i)
                preds.append(float(pred[i]*normal[1] + normal[0]))
        mae = float(sum(val_mae_total)/len(val_mae_total))
    mae_val = loss_mae(pred,torch.unsqueeze(batch[1].y, dim=1))
    if loss < lowest_loss and len(preds) != 0:
        lowest_loss = loss
        y_predicted = preds
        torch.save(model.state_dict(), save_path + "model_mix_best_0623_except_pure_metal.pth")
    return loss ,lowest_loss,mae_val

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
    lr = trial.suggest_categorical('lr', [1e-5,3e-5,6e-5,1e-4,3e-4,6e-4,1e-3])
    # Search the weight decay factor
    weight_decay = trial.suggest_categorical('weight_decay', [1e-6,1e-5, 1e-4,1e-3, 1e-2, 1e0])
    # Search the network architecture hyperparameters
    cgcnn_num_embeding_characters = trial.suggest_categorical('num_characters', [10, 20, 50, 100 , 200])
    cgcnn_num_conv_layers = trial.suggest_int('num_conv_layers', 1, 5)
    cgcnn_num_linear_layers = trial.suggest_int('num_linear_layers',1,5)
    cgcnn_hidden_size = trial.suggest_categorical('cgcnn_hidden_size', [16, 32, 64, 128, 256 , 512])
    cgcnn_dropout = trial.suggest_categorical('cgcnn_dropout', [0, 0.1, 0.2,0.3, 0.4 ,0.5])
    cgcnn_continous_fea_len =  trial.suggest_categorical('continous_size', [4,8, 16, 32])
    cgcnn_part_weight = trial.suggest_categorical('cgcnn_weight', [8, 16, 32, 64, 128])
    
    mlp_num_linear_layers = trial.suggest_int('mlp_num_linear_layers',1,10)
    mlp_hidden_size = trial.suggest_categorical('mlp_hidden_size', [16, 32, 64, 128, 256 , 512])
    mlp_p = trial.suggest_categorical('mlp_dropout', [0, 0.1, 0.2,0.3, 0.4 ,0.5])
    mlp_embedding_p = trial.suggest_categorical('mlp_embedding_dropout', [0, 0.1, 0.2,0.3, 0.4 ,0.5])    
    mlp_part_weight = trial.suggest_categorical('mlp_weight', [8,16, 32, 64, 128])
    
    mix_num_linear_layers = trial.suggest_int('mix_num_linear_layers',0,6)
    mix_p = trial.suggest_categorical('mix_dropout', [0, 0.1, 0.2,0.3, 0.4 ,0.5])
    mix_hidden_size = trial.suggest_categorical('mix_hidden_size', [16, 32, 64, 128, 256 , 512])
    
    class CGCNN_pyg(torch.nn.Module):
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
        
        """
        def __init__(
                self,
               orig_atom_fea_len = 10,
               embbeding_fea_len = int(cgcnn_num_embeding_characters/10),
               atom_fea_len = cgcnn_num_embeding_characters,
               orig_edge_fea_len = 41,
               edge_fea_len =16,
               h_fea_len= cgcnn_hidden_size,
               n_conv = cgcnn_num_conv_layers,
               n_h= cgcnn_num_linear_layers,
               continous_fea_len =  cgcnn_continous_fea_len,
               out_size = cgcnn_part_weight,
               classification = True):
            super(CGCNN_pyg,self).__init__()
            torch.manual_seed(144)
            
            self.embedding = nn.Linear( 10, atom_fea_len,bias = False)
            self.continous_linear = nn.Linear( 3, continous_fea_len,bias = True)
            self.embedding_edge = nn.Linear( 41, edge_fea_len,bias = False)
            self.convs = nn.ModuleList([CGConv([atom_fea_len + continous_fea_len   , atom_fea_len + continous_fea_len ], dim = edge_fea_len, batch_norm = True)
                                        for _ in range(n_conv)])
            self.softpluses1 = nn.ModuleList([nn.Softplus()
                                         for _ in range(n_conv)])
            self.conv_to_fc = nn.Linear(atom_fea_len + continous_fea_len , h_fea_len)
            self.conv_to_fc_softplus = nn.Softplus()
            if n_h > 1:
                self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                          for _ in range(n_h-1)])
                self.softpluses2 = nn.ModuleList([nn.Softplus()
                                                 for _ in range(n_h-1)])
           
            self.fc_out = nn.Linear(h_fea_len,out_size)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(p = cgcnn_dropout)
        
        def forward(self ,x ,edge_index ,edge_attr ,batch_index):
            r"""
            """
            atom_fea = self.embedding(x[:, :10])
            continous_fea = self.continous_linear(x[:,-3:])
            fea_all = torch.cat((atom_fea,continous_fea),dim = 1)
            edge_attr = self.embedding_edge(edge_attr)
            for conv_func, softplus in zip(self.convs, self.softpluses1):
                    atom_fea = softplus(conv_func(fea_all, edge_index , edge_attr))
            crys_fea = gmp(atom_fea, batch_index)
            crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
            crys_fea = self.conv_to_fc_softplus(crys_fea)
            crys_fea = self.dropout(crys_fea)
            if hasattr(self, 'fcs') and hasattr(self, 'softpluses2'):
                for fc, softplus in zip(self.fcs, self.softpluses2):
                    crys_fea = softplus(fc(crys_fea))
            out = self.fc_out(crys_fea)
            out = self.logsoftmax(out)
            return out
    
    class mlp(nn.Module):
        def __init__(self,
                     hidden_size = mlp_hidden_size,
                     output_size = mlp_part_weight,
                     p = mlp_p,
                     p1 = mlp_embedding_p):#or 0.3, which gives less overfitting
            super().__init__()
            torch.manual_seed(42)
            self.embedding_a = torch.nn.Embedding(350, 8)
            self.flatten = torch.nn.Flatten(start_dim = 1,end_dim = -1)
            self.linearA = torch.nn.Linear(16,20)
            self.reluA = torch.nn.ReLU()
            self.dropoutA = nn.Dropout(p1)
    #        self.BN1 = nn.BatchNorm1d(4)
    #        self.BN2 = nn.BatchNorm1d(hidden_size)

            self.embedding_b = torch.nn.Embedding(10, 8)
            self.linearB = torch.nn.Linear(8,8)
            self.reluB = torch.nn.ReLU()
            self.dropoutB = nn.Dropout(p1)
            
            self.embedding_c = torch.nn.Embedding(2, 8)
            self.linearC = torch.nn.Linear(16,8)
            self.reluC = torch.nn.ReLU()
            self.dropoutC = nn.Dropout(p1)
            
            self.linearE = torch.nn.Linear(23,60)#Xie's embedding directly uses nn.Linear
            self.reluE = torch.nn.ReLU()
            self.dropoutE = nn.Dropout(p1)

            self.linear1 = torch.nn.Linear(96, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.lineart1 = torch.nn.Linear(hidden_size,hidden_size)
            self.dropout1 = nn.Dropout(p = 0.6)
            self.relu9 = torch.nn.ReLU()
            
            self.sigmoid1 = torch.nn.Sigmoid()
            self.fcs = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                      for _ in range(mlp_num_linear_layers-1)])
            self.relus = nn.ModuleList([nn.ReLU()
                                             for _ in range(mlp_num_linear_layers-1)])
            self.relus = nn.ModuleList([nn.ReLU()
                                             for _ in range(mlp_num_linear_layers-1)])
            self.dropouts =  nn.ModuleList([nn.Dropout(p)
                                             for _ in range(mlp_num_linear_layers-1)])
    #        self.downsample = downsample
            self.out = torch.nn.Linear(hidden_size, output_size)

        def forward(self, inputA,inputB,inputC, inputE):
            A = self.embedding_a(inputA)
            A = self.flatten(A)
            A = self.linearA(A)
    #        A = self.BN1(A)
            A = self.reluA(A)
            out_A = self.dropoutA(A)
    #        A = BatchNormalization(A)
    
            B = self.embedding_b(inputB)
            B = self.flatten(B)
            B = self.linearB(B)
    #        B = self.BN1(B)
            B = self.reluB(B)
            out_B = self.dropoutB(B)
    #        A = BatchNormalization(A)
            
            C = self.embedding_c(inputC)
            C = self.flatten(C)
            C = self.linearC(C)
    #        C = self.BN1(C)
            C = self.reluC(C)
            out_C = self.dropoutC(C)
    #        C = BatchNormalization(C)

            E = self.linearE(inputE)
    #        B = self.BN1(B)
            E = self.reluE(E)
            out_E = self.dropoutE(E)
            
            x = torch.cat((out_A,out_B,out_C,out_E), dim=1)
            x = self.linear1(x)
            hidden = self.relu1(x)
            hidden = self.dropout1(hidden)
            for fc, relu, dropout in zip(self.fcs, self.relus, self.dropouts):
                hidden = dropout(relu(fc(hidden)))
            hidden = hidden + x
            hidden = self.lineart1(hidden)
            hidden = self.dropout1(hidden)
            hidden = self.relu9(hidden)
            #hidden_all = self.tanh1(hidden_all)
            out = self.out(hidden)
            return out
        
    class MixModel(nn.Module):
        def __init__(self,
                     mlp_output_size = mlp_part_weight, 
                     cgcnn_output_size = cgcnn_part_weight,
                     mix_dropout = mix_p,
                     mix_num_hidden = mix_hidden_size,
                     output_size = 1):
            super().__init__()
            self.mlp_model = mlp(mlp_hidden_size, mlp_output_size)
            self.mlp_model.to(device)
            self.gcn_model = CGCNN_pyg()
            self.gcn_model.to(device)
            self.lineart1 = torch.nn.Linear(mlp_output_size + cgcnn_output_size, mix_num_hidden)
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = nn.Dropout(mix_p)
            
            self.fcs = nn.ModuleList([nn.Linear(mix_num_hidden, mix_num_hidden)
                                             for _ in range(mix_num_linear_layers-1)])
            self.relus = nn.ModuleList([nn.ReLU()
                                             for _ in range(mix_num_linear_layers-1)])
            self.dropouts =  nn.ModuleList([nn.Dropout(mix_p)
                                             for _ in range(mix_num_linear_layers-1)])
            
            self.out = torch.nn.Linear(mix_num_hidden, output_size)#total number of output parameters
            
        def forward(self,data_mlp,data_gcn_x,data_gcn_edge,data_gcn_edge_attr,batch_index):
            mlp_out = self.mlp_model(data_mlp[1],data_mlp[2],data_mlp[3],data_mlp[4])
            gcn_out = self.gcn_model(data_gcn_x, data_gcn_edge,data_gcn_edge_attr,batch_index)
            result_input = torch.cat((mlp_out, gcn_out), 1)
            #result_input = torch.mul(mlp_out , gcn_out)
            hidden_all = self.lineart1(result_input)
            hidden_all = self.dropout1(hidden_all)
            hidden_all = self.relu1(hidden_all)
            for fc, relu, dropout in zip(self.fcs, self.relus, self.dropouts):
                hidden_all = relu(dropout(fc(hidden_all)))
            out = self.out(hidden_all)
            #out = torch.softmax(result_out,dim = 1)
            return out

    # Define the model
    model = MixModel()
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
    print("entering main training loop...")
    for epoch in range(5000):
        loss,mae_train = train(Data_set,optimizer,loss_fn,model)
        losses.append(float(loss))
        train_mae.append(mae_train)
        # Evaluate model performance on the test set
        with torch.no_grad():
            loss2,lowest_loss,mae_val = test(Data_set,lowest_loss,optimizer,loss_fn,model)
            losses2.append(float(loss2))
            val_mae.append(mae_val)
            #print(mae_val)
        trial.report(mae_val, epoch)
        trial_num = trial.number
        if (mae_val < min_mae) and all( float(mae_val) <= _ for _ in mae_main_loop_list) and ((epoch > 100 and trial_num > 0) or (trial_num == 0 and epoch > 200)):
            torch.save(model.state_dict(), save_path + "model_bo_temp-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
        min_mae = min(min_mae,mae_val)
        mae_list.append(mae_val)
        if len(mae_list) > 20:
            mae_list.pop(0)
        min_mae_over_last_20_epochs = min(mae_list)
        if min_mae < min_mae_over_last_20_epochs:
            if all( float(min_mae) <= _ for _ in mae_main_loop_list) and epoch > 100:
                print( '  epoch: {}'.format(epoch))
                torch.save(losses, save_path + "train_loss_temp-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
                torch.save(losses2, save_path + "val_loss_temp-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
                torch.save(train_mae, save_path + "train_mae_temp-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
                torch.save(val_mae, save_path + "val_mae_temp-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
                mae_main_loop_list.append(float(min_mae))
                print('  Best_mae: {}'.format(float(min_mae)))
                parameter = []
                for key, value in trial.params.items():
                    para = []
                    para.append(key)
                    para.append(value)
                    parameter.append(para)
                torch.save(parameter, save_path + "parameter-" + str(trial_num) + "-"+ str(float(mae_val))[:5]+".pt")
            #print(mae_main_loop_list)
            #print(min_mae)
            #print(epoch)
            break
    #draw_loss(losses,losses2)
    #return float(mae_val)
    return float(min_mae)
    
if __name__ == '__main__' :
    #data_path = '../data_set/thermal_conduct/'
    #save_path = "./thermal_conduct/result/0617/"
    data_path = "../data_set/Elasiticity/"
    save_path = "./bulk_modulus/result/0623_excpet_pure_metal/"
    classification = False
    Data_set = data_input(data_path,classification)
    loss_mae = nn.L1Loss()
    for h in range(len(Data_set)):
        if len(Data_set[h][1].edge_index[0]) != len(Data_set[h][1].edge_attr):
            del Data_set[h]
            break
        
    y = []
    for d in Data_set:
        y.append(float(d[1].y))
    normal = []
    normal.append(sum(y)/len(y))
    normal.append(np.std(y))
    
    print('training mean :{0[0]}, training std:{0[1]}'.format(normal))
    Data_set = normalization(Data_set,normal)
    train_loader, test_loader = get_train_val_test_loader(Data_set,batch_size = 2048)
    torch.save(test_loader, save_path + "test_loader_bulk_modulus_0623_except_pure_metal_temp.pt")
    torch.save(normal, save_path + "normalizer_bulk_modulus_0623_except_pure_metal_temp.pt")

    
    mae_main_loop_list = [float('inf')]
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=2000)
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))