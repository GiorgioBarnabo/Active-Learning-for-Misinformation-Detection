import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle as pkl

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
def run_config(base_cfg):
    with wandb.init(project="Misinformation_Detection") as run:  #job_type="train", 
        sweep_cfg = run.config
        cfg = custom_wandb.merge_configs(base_cfg,sweep_cfg)
        cfg = custom_wandb.dotdict(cfg)

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
        
        pipeline_obj = pipeline.Pipeline(cfg,experiment_id)
        pipeline_obj.prepare_pipeline()
        pipeline_obj.run_pipeline()
        pipeline_obj.end_pipeline()

    print("SAVE IN EXPERIMENTS_LIST")
    with open(experiments_list_file, "wb") as f:
      pkl.dump(experiments_list, f)

if __name__ == "__main__":
    config_path="../cfg"
    config_name="sweep_config"
    with open(os.path.join(config_path,config_name+".yaml"), 'r') as f:
        cfg = yaml.safe_load(f)
    #print(cfg)

    cfg = custom_wandb.dotdict(cfg)

    base_config, sweep_config = custom_wandb.divide_sweep(cfg)
    #print("BC",base_config)
    #print("SC",sweep_config)

    sweep_id = wandb.sweep(sweep_config)#, project="Controversy_Detection")
    wandb.agent(sweep_id, function=lambda : run_config(base_config), count=100)
