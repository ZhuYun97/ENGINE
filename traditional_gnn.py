import torch
from torch_geometric.loader import DataLoader
import torch_geometric
from tqdm import tqdm
import numpy as np
import yaml 
from yaml import SafeLoader

from utils.args import Arguments
from utils.sampling import subsampling
from data.load import load_data
from models import load_model


def train_subgraph(model, optimizer, criterion, config, train_loader, val_loader, test_loader, device):
    if config.earlystop:
        cnt = 0
        patience = config.patience
        best_val = 0
        best_test_fromval = 0
        
    for epoch in tqdm(range(config.epochs)):
        model.train()
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model.forward_subgraph(batch.x, batch.edge_index, batch.batch, batch.root_n_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        if config.earlystop:
            val_acc = eval_subgraph(model, val_loader, device)
            if val_acc > best_val:
                cnt = 0
                best_test_fromval = eval_subgraph(model, test_loader, device)
                best_val = val_acc
            else:
                cnt += 1
                if cnt >= patience:
                    print(f'early stop at epoch {epoch}')
                    break
    if not config.earlystop:
        best_test_fromval = eval_subgraph(model, test_loader, device)
    return best_test_fromval

def eval_subgraph(model, data_loader, device):
    model.eval()
    
    correct = 0
    total_num = 0
    for batch in data_loader:
        batch = batch.to(device)
        preds = model.forward_subgraph(batch.x, batch.edge_index, batch.batch, batch.root_n_index).argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total_num += batch.y.shape[0]
    acc = correct / total_num
    return acc

def train_fullgraph(model, optimizer, criterion, config, data, device):
    if config.earlystop:
        cnt = 0
        patience = config.patience
        best_val = 0
        best_test_fromval = 0
    model.train()
    
    data = data.to(device)
    for epoch in tqdm(range(config.epochs)):
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if config.earlystop:
            val_acc = eval_fullgraph(model, data, device, config)
            if val_acc > best_val:
                cnt = 0
                best_test_fromval = eval_fullgraph(model, data, device, config, eval="test")
                best_val = val_acc
            else:
                cnt += 1
                if cnt >= patience:
                    print(f'early stop at epoch {epoch}')
                    break
    if not config.earlystop:
        best_test_fromval = eval_fullgraph(model, data, device, config)
    return best_test_fromval


def eval_fullgraph(model, data, device, config, eval="valid"):
    assert eval in ["valid", "test"]
    model.eval()
    data = data.to(device)
    pred = model(data.x, data.edge_index).argmax(dim=1)
    if eval == "test":
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        total = int(data.test_mask.sum())
    else:
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        total = int(data.val_mask.sum())
    acc = int(correct) / total
    return acc    

def train_eval(model, optimizer, criterion, config, data, train_loader, val_loader, test_loader, device):
    if config.subsampling:
        test_acc = train_subgraph(model, optimizer, criterion, config, train_loader, val_loader, test_loader, device)
    else:
        test_acc = train_fullgraph(model, optimizer, criterion, config, data, device)
    return test_acc

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    acc_list = []
    for i in range(5):
        # load data
        data, text, num_classes = load_data(config.dataset, use_text=True, seed=i)
        data.y = data.y.squeeze()
        
        model = load_model(data.x.shape[1], num_classes, config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        train_loader, val_loader, test_loader = None, None, None
        if config.subsampling:
            train_loader, val_loader, test_loader = subsampling(data, config, sampler=config.sampler)
        test_acc = train_eval(model, optimizer, criterion, config, data, train_loader, val_loader, test_loader, device)
        print(i, test_acc)
        acc_list.append(test_acc)
    
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc*100:.2f}±{final_acc_std*100:.2f}")
        

if __name__ == '__main__':
    args = Arguments().parse_args()
    config = yaml.load(open(args.config), Loader=SafeLoader)
    # combine args and config
    for k, v in config.items():
        args.__setattr__(k, v)
    print(args)
    main(args)