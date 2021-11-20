import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool






import sys
import time


import numpy as np

train_pred = []
train_act = []

test_pred = []
test_act = []

fold = int(sys.argv[1])

st = time.process_time()







path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC')
train_dataset = ZINC(path, subset=True, split='train')
val_dataset = ZINC(path, subset=True, split='val')
test_dataset = ZINC(path, subset=True, split='test')

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

# Compute in-degree histogram over training data.
deg = torch.zeros(5, dtype=torch.long)
for data in train_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())




class UAF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.tensor(1.1, requires_grad=True))
        self.B = torch.nn.Parameter(torch.tensor(-0.01, requires_grad=True))
        self.C = torch.nn.Parameter(torch.tensor(0.00001, requires_grad=True))
        self.D = torch.nn.Parameter(torch.tensor(-0.9, requires_grad=True))
        self.E = torch.nn.Parameter(torch.tensor(0.00001, requires_grad=True))

        self.Softplus = torch.nn.Softplus()
    def forward(self, input):

        return self.Softplus((self.A*(input+self.B)) + (self.C * torch.square(input))) - self.Softplus((self.D*(input-self.B)))  + self.E



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.func =  UAF()


        self.mlp = Sequential(Linear(75, 50), self.func, Linear(50, 25), self.func,
                              Linear(25, 1))




    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = self.func(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam([
    dict(params=model.convs.parameters()),
    dict(params=model.batch_norms.parameters()),
    dict(params=model.mlp.parameters())
], lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)



optimizer2 = torch.optim.Adam([
    dict(params=model.func.parameters())
], lr=0.001)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=300, gamma=1e-10)






def train(epoch):
    model.train()


    train_pred_temp = []
    train_act_temp = []
    first = True



    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        optimizer2.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = (out.squeeze() - data.y).abs().mean()


        pred = out.squeeze()
        if (first):
            train_pred_temp = pred.cpu().detach().numpy()
            train_act_temp = data.y.cpu().detach().numpy()
            first = False
        else:
            train_pred_temp = np.append(train_pred_temp, pred.cpu().detach().numpy())
            train_act_temp = np.append(train_act_temp, data.y.cpu().detach().numpy())



        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        optimizer2.step()


    train_pred.append(train_pred_temp)
    train_act.append(train_act_temp)
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()


    test_pred_temp = []
    test_act_temp = []
    first = True

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()


        pred = out.squeeze()
        if (first):
            test_pred_temp = pred.cpu().detach().numpy()
            test_act_temp = data.y.cpu().detach().numpy()
            first = False
        else:
            test_pred_temp = np.append(test_pred_temp, pred.cpu().detach().numpy())
            test_act_temp = np.append(test_act_temp, data.y.cpu().detach().numpy())


    test_pred.append(test_pred_temp)
    test_act.append(test_act_temp)
    return total_error / len(loader.dataset)



@torch.no_grad()
def test_val(loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)




for epoch in range(1, 601):
    loss = train(epoch)
    val_mae = test_val(val_loader)
    test_mae = test(test_loader)
    if (epoch == 302):
        scheduler.optimizer.param_groups[0]['lr'] = 0.001
    scheduler.step(val_mae)
    scheduler2.step()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')




elapsed_time = time.process_time() - st

np.save("time_" + str(fold), np.array([elapsed_time]))


np.save("train_pred_" + str(fold), train_pred)
np.save("train_act_" + str(fold), train_act)


np.save("test_pred_" + str(fold), test_pred)
np.save("test_act_" + str(fold), test_act)
