import torch
from torch.nn import BatchNorm1d as BN
from torch.nn import Dropout
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import RGCNConv, RGATConv, global_max_pool


def MLP(channels, batch_norm=True):
    if batch_norm:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))
        ])
    else:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU())
            for i in range(1, len(channels))
        ])


class RGCN(torch.nn.Module):
    def __init__(self, n_classes, dropout=False, pool_op='max'):
        super(RGCN, self).__init__()

        self.rgcn_conv1 = RGCNConv(in_channels=3, out_channels=64, num_relations=2, num_bases=64)
        self.rgcn_conv2 = RGCNConv(in_channels=64, out_channels=128, num_relations=2, num_bases=128)

        self.lin1 = MLP([128, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool

        if dropout:
            self.mlp = Seq(
                MLP([1024, 512],batch_norm=True), Dropout(0.5), MLP([512, 256],batch_norm=True),
                Dropout(0.5), Lin(256, n_classes))
        else:
            self.mlp = Seq(
                MLP([1024, 512]), MLP([512, 256]), Lin(256, n_classes))

    def forward(self, data):
        pos, batch, eidx, etypes = data.pos, data.batch, data.edge_index, data.edge_type

        x1 = self.rgcn_conv1(pos, eidx, etypes)
        x2 = self.rgcn_conv2(x1, eidx, etypes)
        x3 = global_max_pool(self.lin1(x2), batch)

        return self.mlp(x3)
