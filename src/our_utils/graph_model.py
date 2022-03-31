import argparse
import time
from tqdm import tqdm
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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from . import gnn_base_models
import os
project_folder = os.path.join('..')


#import os
#os.chdir(
#    "/home/barnabog/Online-Active-Learning-for-Misinformation-Detection/src/gnn_models/"  #ATTENTO_FEDE
#)


#wandb.login()

import sys

# insert at 1, 0 is the script path (or '' in REPL)
import os

'''
os.chdir(
    "/home/barnabog/Online-Active-Learning-for-Misinformation-Detection/src/gnn_models/"
)
'''

#sys.path.append("..")


# sys.path.insert(1, '')

# from our_utils.utils.data_loader import *
#from our_utils.utils.eval_helper import *

#from our_utils.utils import data_loader

#sys.modules["new_utils.graph_utils.data_loader"] = data_loader
#sys.modules["utils.data_loader"] = data_loader


def initialize_graph_model(cfg):
    #print("PARSER")
    parser = argparse.ArgumentParser()
    for k,v in cfg.items():
        parser.add_argument('--'+k, default=v)
    cfg = parser.parse_args()

    model = GNN_Misinfo_Classifier(cfg)

    wandb_logger = pl.loggers.WandbLogger(
        project = "Misinformation_Detection",
        entity = "misinfo_detection",
        #name = str(self.experiment_id), #!!!!!!!WHY?!!!!!!
        save_dir = os.path.join(project_folder,"out","training_logs","wandb"),
    )

    es = pl.callbacks.EarlyStopping(monitor="validation_loss", patience=5)
    checkpointing = pl.callbacks.ModelCheckpoint(
        monitor="validation_loss",
        dirpath=os.path.join(project_folder,"out","models"),
        #filename = str(self.experiment_id), #!!!!!!!WHY?!!!!!!
    )
    
    trainer = pl.Trainer(
        gpus=cfg.gpus_available, #change based on availability
        strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
        # default_root_dir = "../../out/models_checkpoints/",
        max_epochs=cfg.epochs,
        logger=wandb_logger,
        callbacks=[es, checkpointing],
        stochastic_weight_avg=True,
        accumulate_grad_batches=4,
        precision=16,
        log_every_n_steps=50,
    ) 

    return model, trainer

class GNN_Misinfo_Classifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.save_hyperparameters()
        #print("HYPRS SAVED")
        
        if self.cfg.model in ['gcn', 'gat', 'sage']:
            self.model = gnn_base_models.GNN(self.cfg)
        elif self.cfg.model == 'gcnfn':
            self.model = gnn_base_models.Net(self.cfg)
        elif self.cfg.model == 'bigcn':
            self.model = gnn_base_models.BiNet(self.cfg)

        #print("MODEL CREATED")

        # METRICS
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.train_f1_score_micro = torchmetrics.F1Score(average="micro")
        self.val_f1_score_micro = torchmetrics.F1Score(average="micro")
        self.test_f1_score_micro = torchmetrics.F1Score(average="micro")
        self.train_f1_score_macro = torchmetrics.F1Score(average="macro", num_classes=2)
        self.val_f1_score_macro = torchmetrics.F1Score(average="macro", num_classes=2)
        self.test_f1_score_macro = torchmetrics.F1Score(average="macro", num_classes=2)
        self.train_AUC = torchmetrics.AUROC()
        self.val_AUC = torchmetrics.AUROC()
        self.test_AUC = torchmetrics.AUROC()

    def training_step(self, data, batch_idx):
        #print("TRAIN STEP") 
        
        x = self.model(data)
        y = data.y
        train_loss = F.nll_loss(x, y)

        outputs = x.argmax(axis=1)
        outputs_probs = torch.exp(x)[:, 1]

        self.train_acc(y, outputs)
        self.train_f1_score_micro(y, outputs)
        self.train_f1_score_macro(y, outputs)
        self.train_AUC(outputs_probs, y.long())

        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_f1_score_micro", self.train_f1_score_micro, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_f1_score_macro", self.train_f1_score_macro, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_AUC", self.train_AUC, on_step=False, on_epoch=True, prog_bar=False)

        return train_loss

    def forward(self, data):
        #print("FORWARD") 
        x = self.model(data)
        return x

    def validation_step(self, data, batch_idx):
        #print("VAL STEP") 
        
        x = self.model(data)
        y = data.y
        validation_loss = F.nll_loss(x, y)

        outputs = x.argmax(axis=1)
        outputs_probs = torch.exp(x)[:, 1]

        self.val_acc(y, outputs)
        self.val_f1_score_micro(y, outputs)
        self.val_f1_score_macro(y, outputs)
        self.val_AUC(outputs_probs, y.long())

        self.log("validation_loss", validation_loss,on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation_accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("validation_f1_score_micro", self.val_f1_score_micro, on_step=False, on_epoch=True, prog_bar=False)
        self.log("validation_f1_score_macro", self.val_f1_score_macro, on_step=False, on_epoch=True, prog_bar=False)
        self.log("validation_AUC", self.val_AUC, on_step=False, on_epoch=True, prog_bar=False)

        return validation_loss

    def test_step(self, data, batch_idx):
        #print("TEST STEP") 
        
        x = self.model(data)
        y = data.y
        test_loss = F.nll_loss(x, y)
        
        outputs = x.argmax(axis=1)
        outputs_probs = torch.exp(x)[:, 1]

        self.test_acc(y, outputs)
        self.test_f1_score_micro(y, outputs)
        self.test_f1_score_macro(y, outputs)
        self.test_AUC(outputs_probs, y.long())

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_accuracy", self.test_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_f1_score_micro", self.test_f1_score_micro, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_f1_score_macro", self.test_f1_score_macro, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_AUC", self.test_AUC, on_step=False, on_epoch=True, prog_bar=False)

        return test_loss

    def configure_optimizers(self):
        #print("CONF OPTIM") 
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        #print("OPTIM END")
        return optimizer