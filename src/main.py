import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle as pkl

from our_utils import pipeline

import logging

# @hydra.main(config_path="../conf", config_name="config")
# def my_app(config) -> None:
#     print(config.lr)

# if __name__=='__main__':
#     my_app()

#from pipeline import prepare_pipeline

@hydra.main(config_path="../conf", config_name="config")
def my_app(cfg):
# def my_app():
#     cfg = OmegaConf.load("../conf/config.yaml")
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(cfg))
    #print(OmegaConf.to_yaml(cfg))

    print(cfg)

    #Check if configuration already run
    experiments_list_file = os.path.join("..","..","..","..","experiments", "experiments_list.pkl")
    if not os.path.isfile(experiments_list_file):
        print("NOT")
        experiments_list = {cfg:0}
    else:
        with open(experiments_list_file,"rb") as f:
            experiments_list = pkl.load(f)
        
        if cfg in experiments_list: #already run
            print("already run ---> SKIP")
            return None
        else:
            experiments_list[cfg] = len(experiments_list)
        
    print("RUNNING EXPERIMENT")
    pipeline_obj = pipeline.Pipeline(cfg,len(experiments_list))
    pipeline_obj.prepare_pipeline()
    pipeline_obj.run_pipeline()
    pipeline_obj.end_pipeline()

    print("SAVE IN EXPERIMENTS_LIST")
    with open(experiments_list_file, "wb") as f:
      pkl.dump(experiments_list, f)
    
    #return 0 #needed for Optimization Sweepers

if __name__ == "__main__":
    my_app()
