import torch
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DenseSAGEConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GATConv, dense_diff_pool
import copy as cp
from torch_scatter import scatter_mean
from math import ceil

####################################### GCN, GAT, SAGE ##############################################

class GNN(torch.nn.Module):
    def __init__(
        self,
        args,
    ):
        super(GNN, self).__init__()

        self.concat = args.concat
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.model = args.model

        if self.model == "gcn":
            self.conv1 = GCNConv(self.num_features, self.nhid)
        elif self.model == "sage":
            self.conv1 = SAGEConv(self.num_features, self.nhid)
        elif self.model == "gat":
            self.conv1 = GATConv(self.num_features, self.nhid)

        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y

        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = gmp(x, batch)

        if self.concat:
            news = torch.stack(
                [
                    data.x[(data.batch == idx).nonzero().squeeze()[0]]
                    for idx in range(data.num_graphs)
                ]
            )
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))

        x = F.log_softmax(self.lin2(x), dim=-1)

        return x

####################################### GCNFN ##############################################

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        self.args = args
        self.concat = args.concat
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        
        self.conv1 = GATConv(self.num_features, self.nhid * 2)
        self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)
        self.fc1 = Linear(self.nhid * 2, self.nhid)
        
        if self.concat:
            self.fc0 = Linear(self.num_features, self.nhid)
            self.fc1 = Linear(self.nhid * 2, self.nhid)

        self.fc2 = Linear(self.nhid, self.num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        if self.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.fc0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.fc1(x))

        x = F.log_softmax(self.fc2(x), dim=-1)

        return x

####################################### BiGCN ##############################################

class TDrumorGCN(torch.nn.Module):
	def __init__(self, in_feats, hid_feats, out_feats):
		super(TDrumorGCN, self).__init__()
		self.conv1 = GCNConv(in_feats, hid_feats)
		self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x1 = cp.copy(x.float())
		x = self.conv1(x, edge_index)
		x2 = cp.copy(x)
		rootindex = data.root_index
		root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
		batch_size = max(data.batch) + 1

		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x1[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1)

		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x2[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1)
		x = scatter_mean(x, data.batch, dim=0)

		return x


class BUrumorGCN(torch.nn.Module):
	def __init__(self, in_feats, hid_feats, out_feats):
		super(BUrumorGCN, self).__init__()
		self.conv1 = GCNConv(in_feats, hid_feats)
		self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

	def forward(self, data):
		x, edge_index = data.x, data.BU_edge_index
		x1 = cp.copy(x.float())
		x = self.conv1(x, edge_index)
		x2 = cp.copy(x)

		rootindex = data.root_index
		root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
		batch_size = max(data.batch) + 1
		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x1[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1)

		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x2[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1)

		x = scatter_mean(x, data.batch, dim=0)
		return x


class BiNet(torch.nn.Module):
    def __init__(self, args):
        super(BiNet, self).__init__()
        
        self.args = args
        self.in_feats = args.num_features
        self.hid_feats = args.nhid
        self.out_feats = args.nhid

        self.TDrumorGCN = TDrumorGCN(self.in_feats, self.hid_feats, self.out_feats)
        self.BUrumorGCN = BUrumorGCN(self.in_feats, self.hid_feats, self.out_feats)
        self.fc = torch.nn.Linear((self.out_feats+self.hid_feats) * 2, 2)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((TD_x, BU_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


####################################### GNNCL ##############################################

class GNN_CL(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels,
				 normalize=False, lin=True):
		super(GNN_CL, self).__init__()
		self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
		self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
		self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
		self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
		self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
		self.bn3 = torch.nn.BatchNorm1d(out_channels)

		if lin is True:
			self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
									   out_channels)
		else:
			self.lin = None

	def bn(self, i, x):
		batch_size, num_nodes, num_channels = x.size()

		x = x.view(-1, num_channels)
		x = getattr(self, 'bn{}'.format(i))(x)
		x = x.view(batch_size, num_nodes, num_channels)
		return x

	def forward(self, x, adj, mask=None):
		batch_size, num_nodes, in_channels = x.size()

		x0 = x
		x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
		x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
		x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

		x = torch.cat([x1, x2, x3], dim=-1)

		if self.lin is not None:
			x = F.relu(self.lin(x))

		return x


class Net_Cl(torch.nn.Module):
    def __init__(self, args):
        super(Net_Cl, self).__init__()

        self.args = args
        self.max_nodes = self.args.max_nodes
        self.in_channels = 3
        self.num_classes = 6

        self.num_nodes = ceil(0.25 * self.max_nodes)
        self.gnn1_pool = GNN_CL(self.in_channels, 64, self.num_nodes)
        self.gnn1_embed = GNN_CL(self.in_channels, 64, 64, lin=False)

        self.num_nodes = ceil(0.25 * self.num_nodes)
        self.gnn2_pool = GNN_CL(3 * 64, 64, self.num_nodes)
        self.gnn2_embed = GNN_CL(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN_CL(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, self.num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)#, l1 + l2, e1 + e2

####################################### TRANSFORMATIONS ##############################################


class DropEdge:
	def __init__(self, tddroprate, budroprate):
		"""
		Drop edge operation from BiGCN (Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks)
		1) Generate TD and BU edge indices
		2) Drop out edges
		Code from https://github.com/TianBian95/BiGCN/blob/master/Process/dataset.py
		"""
		self.tddroprate = tddroprate
		self.budroprate = budroprate

	def __call__(self, data):
		edge_index = data.edge_index

		if self.tddroprate > 0:
			row = list(edge_index[0])
			col = list(edge_index[1])
			length = len(row)
			poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
			poslist = sorted(poslist)
			row = list(np.array(row)[poslist])
			col = list(np.array(col)[poslist])
			new_edgeindex = [row, col]
		else:
			new_edgeindex = edge_index

		burow = list(edge_index[1])
		bucol = list(edge_index[0])
		if self.budroprate > 0:
			length = len(burow)
			poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
			poslist = sorted(poslist)
			row = list(np.array(burow)[poslist])
			col = list(np.array(bucol)[poslist])
			bunew_edgeindex = [row, col]
		else:
			bunew_edgeindex = [burow, bucol]

		data.edge_index = torch.LongTensor(new_edgeindex)
		data.BU_edge_index = torch.LongTensor(bunew_edgeindex)
		data.root = torch.FloatTensor(data.x[0])
		data.root_index = torch.LongTensor([0])

		return data