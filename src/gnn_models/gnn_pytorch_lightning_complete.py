import argparse
import pickle
import wandb
import sys
import os

import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
import torch_geometric.transforms as T

#print(os.getcwd())

wandb.login()

os.chdir(
    "/home/barnabog/Online-Active-Learning-for-Misinformation-Detection/src/gnn_models/"  #ATTENTO_FEDE
)
sys.path.append("..")
# sys.path.insert(1, '')

# from our_utils.utils.data_loader import *
from our_utils.utils.eval_helper import *

from our_utils.utils import data_loader
import our_utils.utils
from gnn_models.gnn_base_models import *

sys.modules["new_utils.graph_utils.data_loader"] = data_loader  #ATTENTO_FEDE
sys.modules["utils.data_loader"] = data_loader #ATTENTO_FEDE
sys.modules["utils"] = our_utils.utils #ATTENTO_FEDE
sys.modules["new_utils"] = our_utils.utils #ATTENTO_FEDE

####################################### PyTorch Wrapper For All Our GNN Models ###############################################

class GNN_Misinfo_Classifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.save_hyperparameters()
        if self.args.model in ['gcn', 'gat', 'sage']:
            self.model = GNN(self.args)
        elif self.args.model == 'gcnfn':
            self.model = Net(self.args)
        elif self.args.model == 'bigcn':
            self.model = BiNet(self.args)

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
        x = self.model(data)
        return x

    def validation_step(self, data, batch_idx):
        
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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        return optimizer

####################################### Hyperparameters Definition ###############################################

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="politifact", help="[politifact, gossipcop, condor]")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
parser.add_argument("--nhid", type=int, default=128, help="hidden size")
parser.add_argument("--concat", type=bool, default=True, help="whether concat news embedding and graph embedding")
parser.add_argument("--model", type=str, default="sage", help="model type, [gcn, gat, sage]")
args = parser.parse_args()

####################################### Datasets & Dataloaders ###############################################

args.model = "gcn" # "gcn", "gat", "sage", "gcnfn", "bigcn"
args.dataset = "politifact"  # "politifact", "gossipcop", "condor"
workers_available = 3  #ATTENTO_FEDE
gpus_available = [3] #ATTENTO_FEDE

with open(
    "../../data/{}/train_val_test_graphs/training_graph.pickle".format(args.dataset), "rb"
) as f:
    training_set = pickle.load(f)
with open(
    "../../data/{}/train_val_test_graphs/training_graph.pickle".format(args.dataset), "rb"
) as f:
    validation_set = pickle.load(f)
with open("../../data/{}/train_val_test_graphs/training_graph.pickle".format(args.dataset), "rb") as f:
    test_set = pickle.load(f)

if args.model == "bigcn":  #ATTENTO_FEDE
    args.TDdroprate = 0.2
    args.BUdroprate = 0.2
    transformer = DropEdge(args.TDdroprate, args.BUdroprate)
    new_datasets = []
    for dataset in [training_set, validation_set, test_set]:
        new_dataset = []
        for graph in dataset:
            new_dataset.append(transformer(graph))
        new_datasets.append(new_dataset)
    training_set = new_datasets[0]
    validation_set = new_datasets[1]
    test_set = new_datasets[2] 

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=workers_available, pin_memory=True) #ATTENTO_FEDE
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=workers_available, pin_memory=True) #ATTENTO_FEDE
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=workers_available, pin_memory=True) #ATTENTO_FEDE

####################################### Model Initialization & Training - SINGLE RUN ###############################################

args.num_classes = 2
args.num_features = training_set[0].num_features

name_experiment = "prova"

misinformation_classifer = GNN_Misinfo_Classifier(args)
wandb_logger = WandbLogger(
    project="Misinformation_Detection",
    name=name_experiment,
    save_dir="../../out/training_logs/",
)
es = EarlyStopping(monitor="validation_loss", patience=5)
checkpointing = ModelCheckpoint(
    monitor="validation_loss",
    dirpath="../../out/models/",
    filename=name_experiment,
)

trainer = pl.Trainer(
    gpus=gpus_available, #change based on availability
    strategy=DDPPlugin(find_unused_parameters=False),
    # default_root_dir = "../../out/models_checkpoints/",
    max_epochs=1,
    logger=wandb_logger,
    callbacks=[es, checkpointing],
    stochastic_weight_avg=True,
    accumulate_grad_batches=4,
    precision=16,
    log_every_n_steps=50,
)

trainer.fit(
    misinformation_classifer, train_loader, val_loader
)  # testing_loader
trainer.test(misinformation_classifer, test_loader, ckpt_path="best")
wandb.finish()

####################################### Model Initialization & Training - ITERATIVE RUN ###############################################

'''
workers_available = 3  #ATTENTO_FEDE
gpus_available = [3] #ATTENTO_FEDE

for dataset in ["politifact", "gossipcop", "condor"]:
    for model in ["gcn", "gat", "sage", "gcnfn", "bigcn"]:

        args.dataset = dataset
        args.model = model

        with open("../../data/{}/train_val_test_graphs/training_graph.pickle".format(dataset), "rb") as f:
            training_set = pickle.load(f)
        with open("../../data/{}/train_val_test_graphs/validation_graph.pickle".format(dataset), "rb") as f:
            validation_set = pickle.load(f)
        with open("../../data/{}/train_val_test_graphs/test_graph.pickle".format(dataset), "rb") as f:
            test_set = pickle.load(f)

        args.num_classes = 2
        args.num_features = training_set[0].num_features
        
        if args.model == "bigcn":  #ATTENTO_FEDE
            args.TDdroprate = 0.2
            args.BUdroprate = 0.2
            transformer = DropEdge(args.TDdroprate, args.BUdroprate)
            new_datasets = []
            for dataset in [training_set, validation_set, test_set]:
                new_dataset = []
                for graph in dataset:
                    new_dataset.append(transformer(graph))
                new_datasets.append(new_dataset)
            training_set = new_datasets[0]
            validation_set = new_datasets[1]
            test_set = new_datasets[2]
        
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=workers_available, pin_memory=True) #ATTENTO_FEDE
        val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=workers_available, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=workers_available, pin_memory=True)

        name_experiment = "{}_{}".format(args.dataset, args.model)

        misinformation_classifer = GNN_Misinfo_Classifier(args)
        wandb_logger = WandbLogger(
            project="Misinformation_Detection",
            name=name_experiment,
            save_dir="../../out/training_logs/",
        )
        es = EarlyStopping(monitor="validation_loss", patience=5)
        checkpointing = ModelCheckpoint(
            monitor="validation_loss",
            dirpath="../../out/models/",
            filename=name_experiment,
        )

        trainer = pl.Trainer(
            gpus=gpus_available, #change based on availability #ATTENTO_FEDE
            strategy=DDPPlugin(find_unused_parameters=False),
            # default_root_dir = "../../out/models_checkpoints/",
            max_epochs=1,
            logger=wandb_logger,
            callbacks=[es, checkpointing],
            stochastic_weight_avg=True,
            accumulate_grad_batches=4,
            precision=16,
            log_every_n_steps=50,
        )

        trainer.fit(
            misinformation_classifer, train_loader, val_loader
        )  # testing_loader
        trainer.test(misinformation_classifer, test_loader, ckpt_path="best")
        wandb.finish()

'''

####################################### PyTorch Lightning: Use Weights From Pre-Trained Model  ###############################################

# wandb.init(project="Misinformation_Detection") # Controversy_Detection - 2 way classification
# wandb.finish()

# trainer = pl.Trainer()
# chk_path = "../../out/models/first_experiment.ckpt"
# model2 = GNN_Misinfo_Classifier.load_from_checkpoint(chk_path)
# results = trainer.test(model2, test_loader)

# print(results)
