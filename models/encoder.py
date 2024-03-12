import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool, GraphNorm, GCN2Conv
from torch.nn import BatchNorm1d, Identity, LayerNorm
import torch.nn as nn

from utils.register import register
# from .conv import *


def get_activation(name: str):
        activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }
        return activations[name]
    

def get_norm(name: str):
    norms = {
        'id': Identity,
        'bn': BatchNorm1d,
        'ln': LayerNorm
    }
    return norms[name]
    
    
@register.encoder_register
class GCN_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(GCN_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GCNConv(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(GCNConv(hidden_size, hidden_size))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(GCNConv(hidden_size, output_dim))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))

        else: # one layer gcn
            self.convs.append(GCNConv(input_dim, output_dim)) 
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.norms[i](self.convs[i](x, edge_index, edge_weight))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
            

@register.encoder_register
class GCNII_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(GCNII_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GCN2Conv(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(GCN2Conv(hidden_size, hidden_size))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(GCN2Conv(hidden_size, output_dim))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))

        else: # one layer gcn
            self.convs.append(GCN2Conv(input_dim, output_dim)) 
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.norms[i](self.convs[i](x, edge_index, edge_weight))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
                

@register.encoder_register
class SAGE_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(SAGE_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(SAGEConv(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(SAGEConv(hidden_size, hidden_size))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(SAGEConv(hidden_size, output_dim))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))
        else: # one layer gcn
            self.convs.append(SAGEConv(input_dim, output_dim)) 
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.norms[i](self.convs[i](x, edge_index, edge_weight))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
                

@register.encoder_register               
class GIN_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(GIN_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        
        self.readout = global_mean_pool
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, hidden_size)))) 
            for i in range(layer_num-2):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, hidden_size))))
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, output_dim))))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))

        else: # one layer gcn
            self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, hidden_size)))) 
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, **kwargs):
        for i in range(self.layer_num):
            x = self.norms[i](self.convs[i](x, edge_index))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        # out_readout = self.readout(x, batch, batch_size)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
                

@register.encoder_register
class GAT_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(GAT_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm
        self.heads = 8

        self.convs = ModuleList()
        self.norms = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GATConv(input_dim, hidden_size, heads=self.heads, dropout=dropout)) 
            for i in range(layer_num-2):
                self.convs.append(GATConv(hidden_size*self.heads, hidden_size, heads=self.heads, dropout=dropout))
            self.convs.append(GATConv(hidden_size*self.heads, output_dim, heads=1, dropout=dropout))
            
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size*self.heads))
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
        else: # one layer gcn
            self.heads=1
            self.convs.append(GATConv(input_dim, output_dim, heads=self.heads, dropout=dropout)) 
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, **kwargs):
        for i in range(self.layer_num):
            x = self.norms[i](self.convs[i](x, edge_index))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()
            

@register.encoder_register
class MLP_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(MLP_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        
        self.readout = global_mean_pool
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(nn.Linear(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(nn.Linear(hidden_size, hidden_size))
            self.convs.append(nn.Linear(hidden_size, output_dim))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))

        else: # one layer gcn
            self.convs.append(nn.Linear(input_dim, output_dim))
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index=None, **kwargs):
        for i in range(self.layer_num):
            x = self.norms[i](self.convs[i](x))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        # out_readout = self.readout(x, batch, batch_size)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()
            
            
@register.encoder_register
class PMLP_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=128, activation="relu", dropout=0.5, norm='id', last_activation=True):
        super(PMLP_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.norm_type = norm

        self.convs = ModuleList()
        self.norms = ModuleList()
        
        self.readout = global_mean_pool
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(nn.Linear(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(nn.Linear(hidden_size, hidden_size))
            self.convs.append(nn.Linear(hidden_size, output_dim))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.norms.append(get_norm(self.norm_type)(hidden_size))
            self.norms.append(get_norm(self.norm_type)(output_dim))

        else: # one layer gcn
            self.convs.append(nn.Linear(input_dim, hidden_size))
            # glorot(self.convs[-1].weight)
            self.norms.append(get_norm(self.norm_type)(output_dim))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, **kwargs):
        for i in range(self.layer_num):
            x = self.convs[i](x)
            if not self.training:
                x = gcn_conv(x, edge_index) 
            x = self.norms[i](x)
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.norms[i].reset_parameters()