import argparse
import torch
from torch import nn
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from . import gnn_base_models
import os
project_folder = os.path.join('..')

import os

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

    es = pl.callbacks.EarlyStopping(monitor="validation_loss", patience=10)  #validation_f1_score_macro / validation_loss
    
    checkpointing = pl.callbacks.ModelCheckpoint(
        monitor="validation_loss",
        dirpath=os.path.join(project_folder,"out","models"),
        mode='min'
    )
    
    trainer = pl.Trainer(
        gpus=cfg.gpus_available, #change based on availability
        #strategy="ddp", #pl.plugins.DDPPlugin(find_unused_parameters=False),
        # default_root_dir = "../.. /out/models_checkpoints/",
        max_epochs=cfg.epochs,
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[es, checkpointing],
        stochastic_weight_avg=True,
        accumulate_grad_batches=2,
        precision=16,
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
        train_loss = F.nll_loss(x, y, weight=self.cfg.loss_weights)

        outputs = x.argmax(axis=1)
        outputs_probs = torch.exp(x)[:, 1]

        self.train_acc(y, outputs)
        self.train_f1_score_micro(y, outputs)
        self.train_f1_score_macro(y, outputs)
        self.train_AUC(outputs_probs, y.long())

        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=False)
        self.log("train_accuracy", self.train_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("train_f1_score_micro", self.train_f1_score_micro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("train_f1_score_macro", self.train_f1_score_macro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("train_AUC", self.train_AUC, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,rank_zero_only=False)

        return train_loss

    def forward(self, data):
        #print("FORWARD") 
        x = self.model(data)
        return x

    def validation_step(self, data, batch_idx):
        #print("VAL STEP") 
        
        x = self.model(data)
        y = data.y
        validation_loss = F.nll_loss(x, y, weight=self.cfg.loss_weights)

        outputs = x.argmax(axis=1)
        outputs_probs = torch.exp(x)[:, 1]

        self.val_acc(y, outputs)
        self.val_f1_score_micro(y, outputs)
        self.val_f1_score_macro(y, outputs)
        self.val_AUC(outputs_probs, y.long())

        self.log("validation_loss", validation_loss,on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=False)
        self.log("validation_accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("validation_f1_score_micro", self.val_f1_score_micro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("validation_f1_score_macro", self.val_f1_score_macro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("validation_AUC", self.val_AUC, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)

        return validation_loss

    def test_step(self, data, batch_idx):
        #print("TEST STEP") 
        
        x = self.model(data)
        y = data.y
        test_loss = F.nll_loss(x, y, weight=self.cfg.loss_weights)
        
        outputs = x.argmax(axis=1)
        outputs_probs = torch.exp(x)[:, 1]

        self.test_acc(y, outputs)
        self.test_f1_score_micro(y, outputs)
        self.test_f1_score_macro(y, outputs)
        self.test_AUC(outputs_probs, y.long())

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("test_accuracy", self.test_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("test_f1_score_micro", self.test_f1_score_micro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("test_f1_score_macro", self.test_f1_score_macro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("test_AUC", self.test_AUC, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)

        return test_loss

    
    #def set_embeddings_hook(self):
        

    def get_output_and_embeddings(self, loader):
        activation = {}
        def getActivation(name):
            def hook(self, input, output):
                activation[name] = output.detach()
            return hook

        if self.cfg.model in ['gcn', 'gat', 'sage']:
            layer = self.model.lin0
        elif self.cfg.model == 'gcnfn':
            layer = self.model.fc0
        elif self.cfg.model == 'bigcn':
            layer = self.model.TDrumorGCN

        h = layer.register_forward_hook(getActivation('embedding'))

        #pred_y = self.model(data)
        #real_y = data.y
        #error = torch.abs(pred_y - real_y)
        
        #print(self.activation['embedding'])

        #outputs = x.argmax(axis=1)
        #outputs_probs = torch.exp(x)[:, 1]
        
        activations_scores = []
        err_log = []
        
        for data in loader:
            out = self.model(data).argmax(axis=1)
            #print(out)
            #print(data.y)
            err = torch.abs(out-data.y)
            #print(err)
            err_log += list(err.cpu().detach().numpy())
            #print(err_log)
            #out_log += list(F.softmax(out, dim=1).cpu().detach().numpy())
            activations_scores.append(activation['embedding'])
        activations_scores = torch.cat(activations_scores,dim=0)
        
        #h.remove()
        #return np.array(out_log)[:,1], activations_scores
        return err_log, activations_scores

    def configure_optimizers(self):
        #print("CONF OPTIM") 
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        #print("OPTIM END")
        return optimizer


# https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
# https://towardsdatascience.com/type-hints-in-python-everything-you-need-to-know-in-5-minutes-24e0bad06d0b
from typing import List
from typing import Optional

class CoreNet(nn.Module):
    def __init__(self, embeddings_size):
        super().__init__()
        layers: List[nn.Module] = []

        input_dim: int = embeddings_size
        for output_dim in [1024,256,64]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim)),
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, 2))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = F.log_softmax(self.layers(data), dim=1)
        return output

class MultiLabelClassifier(pl.LightningModule):
    def __init__(self,config,embeddings_size):
        super().__init__()
        self.cfg = config
        self.model = CoreNet(embeddings_size)
        
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
        #self.save_hyperparameters()
    
    def training_step(self, data, batch_idx):
        x = self.model(data[0])
        y = data[1]#.long()
        train_loss = F.nll_loss(x, y, weight=self.cfg.deep_al_weights)

        outputs = x.argmax(axis=1)
        outputs_probs = torch.exp(x)[:, 1]

        self.train_acc(y, outputs)
        self.train_f1_score_micro(y, outputs)
        self.train_f1_score_macro(y, outputs)
        self.train_AUC(outputs_probs, y)

        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=False)
        self.log("train_accuracy", self.train_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("train_f1_score_micro", self.train_f1_score_micro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("train_f1_score_macro", self.train_f1_score_macro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("train_AUC", self.train_AUC, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,rank_zero_only=False)

        return train_loss

    def forward(self, data):
        x = self.model(data)
        return x
        
    def validation_step(self, data, batch_idx):
        x = self.model(data[0])
        y = data[1]#.long()
        validation_loss = F.nll_loss(x, y, weight=self.cfg.deep_al_weights)

        outputs = x.argmax(axis=1)
        outputs_probs = torch.exp(x)[:, 1]

        self.val_acc(y, outputs)
        self.val_f1_score_micro(y, outputs)
        self.val_f1_score_macro(y, outputs)
        self.val_AUC(outputs_probs, y.long())

        self.log("validation_loss", validation_loss,on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=False)
        self.log("validation_accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("validation_f1_score_micro", self.val_f1_score_micro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("validation_f1_score_macro", self.val_f1_score_macro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("validation_AUC", self.val_AUC, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)

        return validation_loss

    def test_step(self, data, batch_idx):
        x = self.model(data[0])
        y = data[1]#.long()
        test_loss = F.nll_loss(x, y, weight=self.cfg.deep_al_weights)

        outputs = x.argmax(axis=1)
        outputs_probs = torch.exp(x)[:, 1]

        self.test_acc(y, outputs)
        self.test_f1_score_micro(y, outputs)
        self.test_f1_score_macro(y, outputs)
        self.test_AUC(outputs_probs, y.long())

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("test_accuracy", self.test_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("test_f1_score_micro", self.test_f1_score_micro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("test_f1_score_macro", self.test_f1_score_macro, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        self.log("test_AUC", self.test_AUC, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)

        return test_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        return optimizer