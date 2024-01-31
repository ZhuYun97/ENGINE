from models.encoder import GCN_Encoder
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from utils.register import register


@register.model_register
class GNN(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=70, activation="relu", dropout=0.5, norm='id', **kargs):
        super(GNN, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        
        # self.encoder = GCN_Encoder(input_dim, layer_num, hidden_size, activation, dropout, use_bn)
        self.encoder = register.encoders[kargs['encoder']](input_dim, layer_num, hidden_size, hidden_size, activation, dropout, norm, kargs['last_activation'])
        self.classifier = torch.nn.Linear(hidden_size, output_dim)
        # self.classifier = GCNConv(hidden_size, output_dim)
        self.linear_classifier = torch.nn.Linear(hidden_size*2, output_dim)
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        # x = self.classifier(x, edge_index)
        x = self.classifier(x)
        return x
    
    def forward_subgraph(self, x, edge_index, batch, root_n_id, edge_weight=None, **kwargs):
        # x = torch.rand(x.shape, device=x.device)
        # x = torch.ones(x.shape, device=x.device)
        x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)
        x = self.linear_classifier(x) # use linear classifier
        # x = x[root_n_id]
        # x = self.classifier(x)
        return x
    
    def reset_classifier(self):
        # for i in range(self.layer_num):
        #     self.convs[i].reset_parameters()
        #     self.bns[i].reset_parameters()
        # self.classifier.reset_parameters()
        torch.nn.init.xavier_uniform_(self.linear_classifier.weight.data)
        torch.nn.init.constant_(self.linear_classifier.bias.data, 0)
        
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
        