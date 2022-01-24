import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import time, sys, os
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(1, os.getcwd()+'/graph_utils')
import eval_helper

#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#import os
#print(os.getcwd())

#sys.path.insert(1, os.getcwd()+'/graph_utils')
#from graph_utils.data_loader import *
#from graph_utils.eval_helper import *



"""

The GCN, GAT, and GraphSAGE implementation

"""


class Model(torch.nn.Module):
	def __init__(self, args, concat=False):
		super(Model, self).__init__()
		self.args = args
		self.num_features = args.num_features
		self.nhid = args.nhid
		self.num_classes = args.num_classes
		self.dropout_ratio = args.dropout_ratio
		self.model = args.model
		self.concat = concat
		self.multi_gpu = args.multi_gpu
		self.device = args.device
		self.batch_size = args.batch_size
		self.epochs = args.epochs

		if self.model == 'gcn':
			self.conv1 = GCNConv(self.num_features, self.nhid)
		elif self.model == 'sage':
			self.conv1 = SAGEConv(self.num_features, self.nhid)
		elif self.model == 'gat':
			self.conv1 = GATConv(self.num_features, self.nhid)

		if self.concat:
			self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
			self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

		self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		edge_attr = None

		x = F.relu(self.conv1(x, edge_index, edge_attr))
		x = gmp(x, batch)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.lin0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.lin1(x))

		x = F.log_softmax(self.lin2(x), dim=-1)

		return x
	
	@torch.no_grad()
	def predict(self, loader):
		self.eval()
		out_log = []
		for data in loader:
			print("hurray")
			if not self.multi_gpu:
				data = data.to(self.device)
			out = self(data)
			out_log += list(F.softmax(out, dim=1).cpu().detach().numpy())
		return np.array(out_log)[:,1]


	def training_loop(self, train_loader, val_loader, optimizer):
		min_loss = 1e10
		best_val_loss = np.inf
		best_model = None

		patience = 100
		iter_patience = 0

		t = time.time()
		self.train()
		for epoch in tqdm(range(self.epochs)):
			loss_train = 0.0
			out_log = []
			for i, data in enumerate(train_loader):
				optimizer.zero_grad()
				if not self.multi_gpu:
					data = data.to(self.device)
				out = self(data)
				if self.multi_gpu:
					y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
				else:
					y = data.y
				loss = F.nll_loss(out, y)
				loss.backward()
				optimizer.step()
				loss_train += loss.item()
				out_log.append([F.softmax(out, dim=1), y])
			acc_train, _, _, _, recall_train, auc_train, _ = eval_helper.eval_deep(out_log, train_loader)
			[acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader, self)
			
			if loss_val<best_val_loss:
				best_val_loss = loss_val
				best_model = deepcopy(self)
				iter_patience = 0
			else:
				iter_patience += 1
			if iter_patience >= patience:
				break
			
			#val_loss_values.append(loss_val)
			print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
				f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
				f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
				f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')
		
		return best_model

@torch.no_grad()
def compute_test(loader, model, verbose=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	for data in loader:
		if not model.multi_gpu:
			data = data.to(model.device)
		out = model(data)
		if model.multi_gpu:
			y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
		else:
			y = data.y
		if verbose:
			print(F.softmax(out, dim=1).cpu().numpy())
		out_log.append([F.softmax(out, dim=1), y])
		loss_test += F.nll_loss(out, y).item()
	return eval_helper.eval_deep(out_log, loader), loss_test
'''
if __name__ == '__main__':
	# Model training

	min_loss = 1e10
	val_loss_values = []
	best_epoch = 0

	t = time.time()
	model.train()
	for epoch in tqdm(range(args.epochs)):
		loss_train = 0.0
		out_log = []
		for i, data in enumerate(train_loader):
			optimizer.zero_grad()
			if not args.multi_gpu:
				data = data.to(args.device)
			out = model(data)
			if args.multi_gpu:
				y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
			else:
				y = data.y
			loss = F.nll_loss(out, y)
			loss.backward()
			optimizer.step()
			loss_train += loss.item()
			out_log.append([F.softmax(out, dim=1), y])
		acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
		[acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
		print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
			  f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
			  f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
			  f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

	[acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
	print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
		  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
'''