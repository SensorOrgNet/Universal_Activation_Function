import os.path as osp
import sys
import time


import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np

train_pred = []
train_act = []

test_pred = []
test_act = []

fold = int(sys.argv[1])

st = time.process_time()



dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.



ims = []






class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

        self.A = torch.nn.Parameter(torch.tensor(1.1, requires_grad=True))
        self.B = torch.nn.Parameter(torch.tensor(-0.01, requires_grad=True))
        self.C = torch.nn.Parameter(torch.tensor(1e-9, requires_grad=True))
        self.D = torch.nn.Parameter(torch.tensor(-0.9, requires_grad=True))
        self.E = torch.nn.Parameter(torch.tensor(0.00001, requires_grad=True))


    def UAF(self, input):
        ims.append(np.array([self.A.cpu().detach().item(),self.B.cpu().detach().item(),self.C.cpu().detach().item(),self.D.cpu().detach().item(),self.E.cpu().detach().item()]))
        P1 = (self.A*(input+self.B)) + torch.clamp((self.C * torch.square(input)),-100.0,100.0)
        P2 = (self.D*(input-self.B))

        P3 = torch.nn.ReLU()(P1) + torch.log1p(torch.exp(-torch.abs(P1)))
        P4 = torch.nn.ReLU()(P2) + torch.log1p(torch.exp(-torch.abs(P2)))
        return P3 - P4  + self.E

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.UAF(self.lins[0](x))

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.UAF(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6).to(device)
data = data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.convs.parameters(), weight_decay=0.01),
    dict(params=model.lins.parameters(), weight_decay=5e-4)
], lr=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=50,
                                                       min_lr=0.00001)


optimizer2 = torch.optim.Adam([
    dict(params=model.A),
    dict(params=model.B),
    dict(params=model.C, weight_decay=1e5),
    dict(params=model.D),
    dict(params=model.E)
], lr=0.005)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=240, gamma=1e-10)



def train():
    model.train()
    optimizer2.zero_grad()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    train_pred_temp = out[data.train_mask].cpu().detach().numpy()
    train_act_temp = data.y[data.train_mask].cpu().detach().numpy()

    train_pred.append(train_pred_temp)
    train_act.append(train_act_temp)


    loss.backward()
    optimizer.step()
    optimizer2.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred, accs = model(data.x, data.adj_t).argmax(dim=-1), []

    test_pred_temp = pred[data.test_mask].cpu().detach().numpy()
    test_act_temp = data.y[data.test_mask].cpu().detach().numpy()

    test_pred.append(test_pred_temp)
    test_act.append(test_act_temp)

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 1001):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    lr = scheduler.optimizer.param_groups[0]['lr']
    if (epoch == 241):
        scheduler.optimizer.param_groups[0]['lr'] = 0.05
    scheduler.step(-val_acc)
    scheduler2.step()

    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
          f'lr: {lr:.7f}, Test: {tmp_test_acc:.4f}, '
          f'Final Test: {test_acc:.4f}')



elapsed_time = time.process_time() - st

np.save("time_" + str(fold), np.array([elapsed_time]))


np.save("train_pred_" + str(fold), train_pred)
np.save("train_act_" + str(fold), train_act)


np.save("test_pred_" + str(fold), test_pred)
np.save("test_act_" + str(fold), test_act)


ims = np.asarray(ims)
np.save("ims_" + str(fold),ims)
