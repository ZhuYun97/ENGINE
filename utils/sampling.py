
from torch_geometric.loader import DataLoader

from data.sampling import collect_subgraphs, ego_graphs_sampler


def subsampling(data, config, sampler='rw'):
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    if sampler == 'rw':
        train_graphs = collect_subgraphs(train_idx, data, walk_steps=config.walk_steps, restart_ratio=config.restart)
        val_graphs = collect_subgraphs(val_idx, data, walk_steps=config.walk_steps, restart_ratio=config.restart)
        test_graphs = collect_subgraphs(test_idx, data, walk_steps=config.walk_steps, restart_ratio=config.restart)
    elif sampler == 'khop':
        train_graphs = ego_graphs_sampler(train_idx, data, hop=config.k)
        val_graphs = ego_graphs_sampler(val_idx, data, hop=config.k)
        test_graphs = ego_graphs_sampler(test_idx, data, hop=config.k)
    kwargs = {'batch_size': config.batch_size, 'num_workers': 6, 'persistent_workers': True}
    train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
    val_loader = DataLoader(val_graphs, **kwargs)
    test_loader = DataLoader(test_graphs, **kwargs)
    
    return train_loader, val_loader, test_loader