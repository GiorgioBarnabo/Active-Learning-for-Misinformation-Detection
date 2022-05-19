import os
import pickle as pkl
import numpy as np

from our_utils import pipeline
from our_utils import custom_wandb
import wandb
import torch

sweep_config = {
    'method': 'random',  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
    'metric': {  # This is the metric we are interested in maximizing
      'name': 'validation_loss',
      'goal': 'minimize'   
    },
    # Paramters and parameter values we are sweeping across
    'parameters': {
        'dataset': {
            'values': ['gossipcop', 'politifact', 'condor']
        },
        'model': {
            'values': ["gcn", "gat", "sage", "gcnfn"] #"bigcn"
        },
        'AL_method': {
            'values': ["random", "uncertainty-margin"]
        },
    }
}

def run_config():
    with wandb.init(project="Misinformation_Detection") as run:  #job_type="train", 
        cfg = run.config

        run_single_config(cfg)

def run_single_config(cfg):

    cfg.warm_start_years = [np.inf, np.inf]
    cfg.training_years = [2005,2021]
    cfg.batch_size = 64 # politifact: 32 | condor: 64
    cfg.iteration_of_random_warm_start = 5   # politifact: 2 | condor: 5
    cfg.number_AL_iteration = 100  # politifact: 20 | condor: 100
    cfg.tot_num_checked_urls = 1000 # politifact: 100 | condor: 1000
    cfg.retrain_from_scratch = True
    cfg.train_last_samples = np.inf
    cfg.val_last_samples = np.inf
    cfg.add_val_to_train  = False

    cfg.experiment_batch = 3
    cfg.epochs = 100
    cfg.lr = 0.0005
    cfg.weight_decay = 0.01
    cfg.nhid = 128
    cfg.concat = True
    cfg.workers_available = 3
    cfg.gpus_available = [1] #0, 1
    cfg.num_classes = 2 
    cfg.nb_samples = 1
    cfg.loss_weights_val = torch.tensor([1.2791, 1.0000]).to('cuda:{}'.format(cfg.gpus_available[0]))  #condor: 1.2791

    #Check if configuration already run
    # experiments_list_file = os.path.join("..","out","experiments", "experiments_list.pkl")
    # if not os.path.isfile(experiments_list_file):
    #     print("EXPERIMENT FILE NOT EXISTING")
    #     experiment_id = 0
    #     experiments_list = [cfg]
    # else:
    #     with open(experiments_list_file,"rb") as f:
    #         experiments_list = pkl.load(f)
        
    #     try:
    #         experiment_id = experiments_list.index(self.original_cfg)
    #         #if not redo:
    #         print("already run ---> SKIP")
    #         return None
    #     except ValueError:
    #         experiment_id = len(experiments_list)
    #         experiments_list.append(cfg)
        
    print("RUNNING EXPERIMENT")
    print(cfg)
    
    print("INITIALIZE PIPELINE")
    pipeline_obj = pipeline.Pipeline(cfg)

    print("PREPARE PIPELINE")
    pipeline_obj.prepare_pipeline()

    print("RUN PIPELINE")
    pipeline_obj.run_pipeline()

    # print("SAVE IN EXPERIMENTS_LIST")
    # with open(experiments_list_file, "wb") as f:
    #     pkl.dump(experiments_list, f)

# 'gat', 'sage', 'gcn', 'gcnfn'
# 'sage', 'gcn', 'gcnfn', 'gat'
# 'gcn', 'gcnfn', 'gat', 'sage'
# 'gcnfn', 'gat', 'sage', 'gcn'

# 'gat', 'sage', 'gcn', 'gcnfn'
# 'gcnfn', 'gcn', 'sage', 'gat'

AL_1 = 'deep-discriminator uncertainty-margin'
AL_2 = 'uncertainty-margin' #ok
AL_3 = 'uncertainty-margin+diversity' #ok
AL_4 = 'random' #ok
AL_5 = 'deep-discriminator'
AL_6 = 'deep-discriminator+diversity'
AL_7 = 'deep-adversarial' 
AL_8 = 'deep-adversarial+diversity'

block_1 = ['uncertainty-margin', 'deep-discriminator+diversity', 'deep-discriminator']
block_2 = ['deep-adversarial', 'deep-adversarial+diversity', 'random']
block_3 = ['uncertainty-margin+diversity', 'deep-discriminator uncertainty-margin']

if __name__ == "__main__":
    for dataset in ['condor']:
        for model in ['gat', 'sage', 'gat', 'sage', 'gcn']:
            for AL in block_3:  
                cfg = {
                    'dataset': dataset,
                    'model': model,
                    'AL_method': AL}   
                cfg = custom_wandb.dotdict(cfg)
                run_single_config(cfg)

