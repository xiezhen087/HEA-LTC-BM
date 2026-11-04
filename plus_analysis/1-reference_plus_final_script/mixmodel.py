# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 00:53:20 2024

@author: Realunknown
"""

from torch_geometric.nn import CGConv
from torch_geometric.nn import global_mean_pool as gmp
import torch
import torch.nn as nn

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
    def __init__(self, input_size =7, hidden_size=128, output_size=80 ,p = 0.3,p1 = 0.3):#or 0.3, which shows less overfitting
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

        self.linearE = torch.nn.Linear(23,60)#xie's embedding directly uses nn.Linear
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