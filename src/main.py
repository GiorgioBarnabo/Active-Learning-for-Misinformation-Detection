import os
import pickle as pkl
import numpy as np

from our_utils import pipeline
from our_utils import custom_wandb

import logging
import wandb
import yaml

# @hydra.main(config_path="../conf", config_name="config")
# def my_app(config) -> None:
#     print(config.lr)

# if __name__=='__main__':
#     my_app()

#from pipeline import prepare_pipeline

#@hydra.main(config_path="../conf", config_name="config")


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
    with wandb.init(project="Misinformation_Detection", job_type="train") as run:  #job_type="train", 
        cfg = run.config

        run_single_config(cfg)
        

def run_single_config(cfg):
    cfg.warm_start_years = [np.inf, np.inf]
    cfg.training_years = [2005,2021]
    cfg.batch_size = 32
    cfg.number_AL_iteration = 8
    cfg.tot_num_checked_urls = 200
    cfg.retrain_from_scratch = True
    cfg.train_last_samples = np.inf
    cfg.val_last_samples = np.inf
    cfg.add_val_to_train  = False
    
    cfg.epochs = 30
    cfg.lr = 0.005
    cfg.weight_decay = 0.01
    cfg.nhid = 128
    cfg.concat = True
    cfg.workers_available = 4
    cfg.gpus_available = [2]
    cfg.num_classes = 2 
    cfg.nb_samples = 1

    #Check if configuration already run
    experiments_list_file = os.path.join("..","out","experiments", "experiments_list.pkl")
    if not os.path.isfile(experiments_list_file):
        print("EXPERIMENT FILE NOT EXISTING")
        experiment_id = 0
        experiments_list = [cfg]
    else:
        with open(experiments_list_file,"rb") as f:
            experiments_list = pkl.load(f)
        
        try:
            experiment_id = experiments_list.index(self.original_cfg)
            #if not redo:
            print("already run ---> SKIP")
            return None
        except ValueError:
            experiment_id = len(experiments_list)
            experiments_list.append(cfg)
        
    print("RUNNING EXPERIMENT")
    print(cfg)
    
    print("INITIALIZE PIPELINE")
    pipeline_obj = pipeline.Pipeline(cfg, experiment_id)

    print("PREPARE PIPELINE")
    pipeline_obj.prepare_pipeline()

    print("RUN PIPELINE")
    pipeline_obj.run_pipeline()

    #print("END PIPELINE")
    #pipeline_obj.end_pipeline()

    print("SAVE IN EXPERIMENTS_LIST")
    with open(experiments_list_file, "wb") as f:
        pkl.dump(experiments_list, f)

if __name__ == "__main__":
    #config_path="../cfg"
    #config_name="sweep_config_new"
    # with open(os.path.join(config_path,config_name+".yaml"), 'r') as f:
    #     cfg = yaml.safe_load(f)
    #print(cfg)

    # cfg = custom_wandb.dotdict(cfg)

    # base_config, sweep_config = custom_wandb.divide_sweep(cfg)
    #print("BC",base_config)
    #print("SC",sweep_config)

    
    #os.environ["WANDB_CONSOLE"] = "off" ####TO AVOID ValueError('signal only works in main thread')???

    #SWEEP:
    #sweep_id = wandb.sweep(sweep_config)#, project="Controversy_Detection")
    #wandb.agent(sweep_id, function=run_config, count=1)

    #SINGLE:
    
    for dataset in ["condor"]:
        for AL in ["uncertainty-margin", "random"]:
    
            cfg = {
                'dataset': dataset,
                'model': "gcn",
                'AL_method':AL}
                
            cfg = custom_wandb.dotdict(cfg)
            run_single_config(cfg)

