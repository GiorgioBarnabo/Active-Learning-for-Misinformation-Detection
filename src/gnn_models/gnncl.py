import argparse
from tqdm import tqdm
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch.utils.data import random_split
from copy import deepcopy

import sys
# insert at 1, 0 is the script path (or '' in REPL)
import os
print(os.getcwd())
sys.path.insert(1, '')

from utils.data_loader import *
from utils.eval_helper import *

from utils.data_loader import *
from utils.eval_helper import *

project_folder = os.path.join('../../../')

"""

The GNN-CL is implemented using DiffPool as the graph encoder and profile feature as the node feature 

Paper: Graph Neural Networks with Continual Learning for Fake News Detection from Social Media
Link: https://arxiv.org/pdf/2007.03316.pdf

"""


class GNN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels,
				 normalize=False, lin=True):
		super(GNN, self).__init__()
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


class Net(torch.nn.Module):
	def __init__(self, in_channels=3, num_classes=6):
		super(Net, self).__init__()

		num_nodes = ceil(0.25 * max_nodes)
		self.gnn1_pool = GNN(in_channels, 64, num_nodes)
		self.gnn1_embed = GNN(in_channels, 64, 64, lin=False)

		num_nodes = ceil(0.25 * num_nodes)
		self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
		self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

		self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

		self.lin1 = torch.nn.Linear(3 * 64, 64)
		self.lin2 = torch.nn.Linear(64, num_classes)

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
		return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


def train():
	model.train()
	loss_all = 0
	out_log = []
	for data in train_loader:
		data = data.to(device)
		optimizer.zero_grad()
		out, _, _ = model(data.x, data.adj, data.mask)
		out_log.append([F.softmax(out, dim=1), data.y])
		loss = F.nll_loss(out, data.y.view(-1))
		loss.backward()
		loss_all += data.y.size(0) * loss.item()
		optimizer.step()
	return eval_deep(out_log, train_loader), loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
	model.eval()

	loss_test = 0
	out_log = []
	for data in loader:
		data = data.to(device)
		out, _, _ = model(data.x, data.adj, data.mask)
		out_log.append([F.softmax(out, dim=1), data.y])
		loss_test += data.y.size(0) * F.nll_loss(out, data.y.view(-1)).item()
	return eval_deep(out_log, loader), loss_test


parser = argparse.ArgumentParser()
#parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop, condor]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')

args = parser.parse_args()
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
# 	torch.cuda.manual_seed(args.seed)

for data_set in ['condor', 'gossipcop']:  #'politifact'

	args.dataset = data_set

	if args.dataset == 'politifact':
		max_nodes = 500
	else:
		max_nodes = 200 


	dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset,
						transform=T.ToDense(max_nodes), pre_transform=ToUndirected())

	print(args)

	split_ratios = [[0.05, 0.35, 0.60], [0.10, 0.30, 0.60], [0.20, 0.30, 0.50], [0.50, 0.20, 0.30]]
	
	for train_val_test in split_ratios:
		
		split_ratio = train_val_test

		num_training = int(len(dataset) * split_ratio[0])
		num_val = int(len(dataset) * split_ratio[1])
		num_test = len(dataset) - (num_training + num_val)
		training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

		train_loader = DenseDataLoader(training_set, batch_size=args.batch_size, shuffle=True)
		val_loader = DenseDataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
		test_loader = DenseDataLoader(test_set, batch_size=args.batch_size, shuffle=False)

		f = open(os.path.join(project_folder, 'src', 'models', 'GNN-FakeNews', 'results', 'gnncl_results.txt'), 'a')
		info = '{}={}\tepochs={}\ttrain={:.2f}\tval={:.2f}\ttest={:.2f}\n'.format(args.dataset, len(dataset), args.epochs, split_ratio[0], split_ratio[1], split_ratio[2])
		f.write(info)
		f.close()

		for i in range(5):

			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			model = Net(in_channels=dataset.num_features, num_classes=dataset.num_classes).to(device)
			optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

			best_val_loss = np.inf
			best_model = None
			patient = 0

			for epoch in tqdm(range(args.epochs)):
				[acc_train, _, _, _, recall_train, auc_train, _], loss_train = train()
				[acc_val, _, _, _, recall_val, auc_val, _], loss_val = test(val_loader)
				
				if loss_val<best_val_loss:
					best_val_loss = loss_val
					best_model = deepcopy(model)
					patient = 0
				else:
					patient += 1
					if patient >=10:
						break
				
				print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
					f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
					f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
					f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')


			model = best_model

			'''
			[acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
			print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f},'
					f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
			'''

			rs = new_gnncl_compute_test(model, test_loader, args, verbose=False)

			f = open(os.path.join(project_folder, 'src', 'models', 'GNN-FakeNews', 'results', 'gnncl_results.txt'), 'a')
			info = '{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(args.dataset, rs[0], rs[1], rs[2], rs[3], rs[4], rs[5], rs[6])
			f.write(info)
			f.close()
			print(info)