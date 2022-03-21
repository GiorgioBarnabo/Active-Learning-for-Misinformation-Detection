import argparse
import time
from tqdm import tqdm
from copy import deepcopy
import pickle
import numpy as np
from sklearn import metrics

import torch
from torch import nn
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

wandb.login()

import sys
# insert at 1, 0 is the script path (or '' in REPL)
import os
print(os.getcwd())

os.chdir('/home/barnabog/Online-Active-Learning-for-Misinformation-Detection/src/gnn_models')
sys.path.append('..')
#sys.path.insert(1, '')

#from our_utils.utils.data_loader import *
from our_utils.utils.eval_helper import *

from our_utils.utils import data_loader
sys.modules['new_utils.graph_utils.data_loader'] = data_loader

class GNN_Misinfo_Classifier(pl.LightningModule):
	def __init__(self, args, concat=False):
		super().__init__()
		self.args = args
		self.num_features = args.num_features
		self.nhid = args.nhid
		self.num_classes = args.num_classes
		self.dropout_ratio = args.dropout_ratio
		self.model = args.model
		self.concat = concat
		
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

		# METRICS
		self.train_acc = torchmetrics.Accuracy()
		self.val_acc = torchmetrics.Accuracy()
		self.test_acc = torchmetrics.Accuracy()
		self.train_f1_score_micro = torchmetrics.F1Score(average='micro')
		self.val_f1_score_micro = torchmetrics.F1Score(average='micro')
		self.test_f1_score_micro = torchmetrics.F1Score(average='micro')
		self.train_f1_score_macro = torchmetrics.F1Score(average='macro', num_classes=2)
		self.val_f1_score_macro = torchmetrics.F1Score(average='macro', num_classes=2)
		self.test_f1_score_macro = torchmetrics.F1Score(average='macro', num_classes=2)
		self.train_AUC = torchmetrics.AUROC()
		self.val_AUC = torchmetrics.AUROC()
		self.test_AUC = torchmetrics.AUROC()


	def training_step(self, data, batch_idx):
		# training_step defines the train loop. It is independent of forward
		x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
		edge_attr = None
		x = F.relu(self.conv1(x, edge_index, edge_attr))
		x = gmp(x, batch)
		
		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.lin0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.lin1(x))
		
		x = F.log_softmax(self.lin2(x), dim=-1)
		train_loss = F.nll_loss(x, y)
		outputs = x.argmax(axis=1)
		outputs_probs = torch.exp(x)[:, 1]

		self.train_acc(y, outputs)
		self.train_f1_score_micro(y, outputs)
		self.train_f1_score_macro(y, outputs)
		self.train_AUC(outputs_probs, y.long())

		self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)
		self.log("train_accuracy", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
		self.log("train_f1_score_micro", self.train_f1_score_micro, on_step=False, on_epoch=True, prog_bar=True)
		self.log("train_f1_score_macro", self.train_f1_score_macro, on_step=False, on_epoch=True, prog_bar=True)
		self.log("train_AUC", self.train_AUC, on_step=False, on_epoch=True, prog_bar=True)
		
		return train_loss 
	
	
	def forward(self, data):
        # in lightning, forward defines the prediction/inference actions
		x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y

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

	def validation_step(self, data, batch_idx):
		# training_step defines the train loop. It is independent of forward
		x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
		edge_attr = None
		x = F.relu(self.conv1(x, edge_index, edge_attr))
		x = gmp(x, batch)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.lin0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.lin1(x))
		
		x = F.log_softmax(self.lin2(x), dim=-1)
		validation_loss = F.nll_loss(x, y)
		outputs = x.argmax(axis=1)
		outputs_probs = torch.exp(x)[:, 1]

		self.val_acc(y, outputs)
		self.val_f1_score_micro(y, outputs)
		self.val_f1_score_macro(y, outputs)
		self.val_AUC(outputs_probs, y.long())

		self.log("validation_loss", validation_loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log("validation_accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
		self.log("validation_f1_score_micro", self.val_f1_score_micro, on_step=False, on_epoch=True, prog_bar=True)
		self.log("validation_f1_score_macro", self.val_f1_score_macro, on_step=False, on_epoch=True, prog_bar=True)
		self.log("validation_AUC", self.val_AUC, on_step=False, on_epoch=True, prog_bar=True)
		
		return validation_loss

	def test_step(self, data, batch_idx):

		x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
		edge_attr = None
		x = F.relu(self.conv1(x, edge_index, edge_attr))
		x = gmp(x, batch)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.lin0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.lin1(x))
		
		x = F.log_softmax(self.lin2(x), dim=-1)
		test_loss = F.nll_loss(x, y)
		outputs = x.argmax(axis=1)
		outputs_probs = torch.exp(x)[:, 1]

		self.test_acc(y, outputs)
		self.test_f1_score_micro(y, outputs)
		self.test_f1_score_macro(y, outputs)
		self.test_AUC(outputs_probs, y.long())

		self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log("test_accuracy", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
		self.log("test_f1_score_micro", self.test_f1_score_micro, on_step=False, on_epoch=True, prog_bar=True)
		self.log("test_f1_score_macro", self.test_f1_score_macro, on_step=False, on_epoch=True, prog_bar=True)
		self.log("test_AUC", self.test_AUC, on_step=False, on_epoch=True, prog_bar=True)
		
		return test_loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		return optimizer


parser = argparse.ArgumentParser()

#parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop, condor]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')
parser.add_argument('--model', type=str, default='sage', help='model type, [gcn, gat, sage]')

args = parser.parse_args()

# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
# 	torch.cuda.manual_seed(args.seed)

#dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())

with open("../../data/condor/train_val_test_graphs/training_graph.pickle", 'rb') as f:
	training_set = pickle.load(f)
with open("../../data/condor/train_val_test_graphs/validation_graph.pickle", 'rb') as f:
	validation_set = pickle.load(f)
with open("../../data/condor/train_val_test_graphs/test_graph.pickle", 'rb') as f:
	test_set = pickle.load(f)

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=5) #, num_workers=80)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=5) #, num_workers=80)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=5) #, num_workers=80)

args.num_classes = 2
args.num_features = training_set[0].num_features

misinformation_classifer = GNN_Misinfo_Classifier(args, concat=args.concat)
#wandb.init(project="Misinformation_Detection") # Controversy_Detection - 2 way classification
wandb_logger = WandbLogger(project="Misinformation_Detection")
es = EarlyStopping(monitor="validation_loss", patience=5)
checkpointing = ModelCheckpoint(monitor="validation_loss")
trainer = pl.Trainer(gpus=4, accelerator="ddp", max_epochs=100, logger=wandb_logger, callbacks=[es, checkpointing])
trainer.fit(misinformation_classifer, train_loader, val_loader) #testing_loader
trainer.test(misinformation_classifer, test_loader, ckpt_path="best")
#wandb.finish()


