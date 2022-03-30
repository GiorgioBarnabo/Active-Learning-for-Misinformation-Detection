import sys, os, argparse
sys.path.append('..')

project_folder = os.path.join('..')

import numpy as np

#import our scripts
from . import AL
from . import data_utils
from . import graph_model
from . import gnn_base_models
import wandb


from torch_geometric.loader import DataLoader, DataListLoader
sys.path.append("..")
import pytorch_lightning as pl

#from omegaconf import OmegaConf,open_dict


class Pipeline():
    def __init__(self, cfg, experiment_id):
        self.cfg = cfg
        self.experiment_id = experiment_id

    def prepare_pipeline(self):
        #with open_dict(self.cfg): #needed if trying to set attributes not in config
        #self.graph_args = self.prepare_graph_parser()

        #Change "inf" to np.inf
        # for i,x in enumerate(self.cfg.data_params.warm_start_years):
        #     if x=="inf":
        #         self.cfg.data_params.warm_start_years[i] = np.inf

        # if self.cfg.AL_params.train_last_samples=="inf":
        #     self.cfg.AL_params.train_last_samples = np.inf

        # if self.cfg.AL_params.val_last_samples=="inf":
        #     self.cfg.AL_params.val_last_samples = np.inf

        ####Compute num_urls_k_list:
        if "num_urls_k" not in self.cfg:   #FEDE_WHAT_TO_DO
            if self.cfg.number_AL_iteration>0:
                self.cfg.num_urls_k = int(self.cfg.tot_num_checked_urls//self.cfg.number_AL_iteration)
            else:
                print("TO BE DEFINED ---> divide tot_num_... for number of time-steps?")
                print(error)

        #Set results folder
        self.results_folder = os.path.join(project_folder, 'out', "results",str(self.experiment_id))#, self.cfg.experiment_params.results_set, self.cfg.data_params.dataname)
        #create results_folder not exists
        if not os.path.isdir(self.results_folder):
            os.makedirs(self.results_folder)

        self.data_folder = os.path.join(project_folder, 'data', "graph", self.cfg.dataset)

        #Create WandB logger
        self.wandb_logger = pl.loggers.WandbLogger(
            project = "Misinformation_Detection",
            entity = "misinfo_detection",
            name = str(self.experiment_id), #!!!!!!!WHY?!!!!!!
            save_dir= os.path.join(project_folder,"out","training_logs","wandb"),
        )

        es = pl.callbacks.EarlyStopping(monitor="validation_loss", patience=5)
        checkpointing = pl.callbacks.ModelCheckpoint(
            monitor="validation_loss",
            dirpath=os.path.join(project_folder,"out","models"),
            filename = str(self.experiment_id), #!!!!!!!WHY?!!!!!!
        )
        
        self.trainer = pl.Trainer(
            gpus=self.cfg.gpus_available, #change based on availability
            strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
            # default_root_dir = "../../out/models_checkpoints/",
            max_epochs=self.cfg.epochs,
            logger=self.wandb_logger,
            callbacks=[es, checkpointing],
            stochastic_weight_avg=True,
            accumulate_grad_batches=4,
            precision=16,
            log_every_n_steps=50,
        )

    def run_pipeline(self):
        all_train_data, all_val_data, all_test_data = data_utils.load_graph_data(self.data_folder) #to load all data
        print("all_train_data.keys", all_train_data.keys())
        print("all_val_data.keys", all_val_data.keys())
        print("all_test_data.keys", all_test_data.keys())

        if self.cfg.model == "bigcn":
            self.cfg.TDdroprate = 0.2
            self.cfg.BUdroprate = 0.2
            transformer = gnn_base_models.DropEdge(self.cfg.TDdroprate, self.cfg.BUdroprate)
            new_datasets = []
            for dataset in [all_train_data, all_val_data, all_test_data]:
                new_dataset = []
                for graph in dataset:
                    new_dataset.append(transformer(graph))
                new_datasets.append(new_dataset)
            all_train_data = new_datasets[0]
            all_val_data = new_datasets[1]
            all_test_data = new_datasets[2] 
        
        first_key = list(all_train_data.keys())[0]
        self.cfg.num_features = all_train_data[first_key][0].num_features #FEDE_WHAT_TO_DO

        #WARM-START PREPARATION
        warm_start_starting_key_id, warm_start_ending_key_id = data_utils.get_warm_start_key_ids(
            all_train_data.keys(), 
            self.cfg.warm_start_years)
        
        print(warm_start_starting_key_id,warm_start_ending_key_id)
        
        doing_warm_start = True

        if warm_start_starting_key_id==len(all_train_data)-1 or warm_start_ending_key_id==0:
            print("!!! NOT DOING WARM START !!!")
            warm_start_starting_key_id = 0
            if self.cfg.number_AL_iteration>0:
                warm_start_ending_key_id = len(all_train_data)
            else:
                warm_start_ending_key_id = 1
            doing_warm_start = False

        #PREPARE ITERATION RANGES
        if self.cfg.number_AL_iteration>0:
            if doing_warm_start:
                iteration_ranges = list(zip([warm_start_starting_key_id]+[warm_start_ending_key_id]*self.cfg.number_AL_iteration,
                                            [warm_start_ending_key_id]+[len(all_train_data)]*self.cfg.number_AL_iteration))
            else:
                iteration_ranges = list(zip([warm_start_starting_key_id]*self.cfg.number_AL_iteration,
                                            [len(all_train_data)]*self.cfg.number_AL_iteration))
        else:
            iteration_ranges = list(zip([warm_start_starting_key_id]+list(range(warm_start_ending_key_id, len(all_train_data))),
                                    range(warm_start_ending_key_id, len(all_train_data)+1)))
        print("IT_RNG:",iteration_ranges)
            
        results_filename = os.path.join(self.results_folder,"rs.npy")#os.path.join(results_folder, '_ALm_' + AL_method + '_k_' + str(num_urls_k) + '.npy')
        all_pos_neg_filename = os.path.join(self.results_folder,"pos_neg.npy")   #FEDE_WHAT_TO_DO
        
        for sample in range(self.cfg.nb_samples): #FEDE_WHAT_TO_DO
            current_data = {"train": None,
                            "val": all_val_data[list(all_val_data.keys())[0]],
                            "test": all_test_data[list(all_test_data.keys())[0]]}

            current_loaders = {"train": None,
                                "val": DataLoader(all_val_data[list(all_val_data.keys())[0]],batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.workers_available, pin_memory=True),
                                "test": DataLoader(all_test_data[list(all_test_data.keys())[0]],batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.workers_available, pin_memory=True)}
            
            #Initialize model
            print("INITIALIZING MODEL")
            model = graph_model.initialize_graph_model(self.cfg)

            #print(current_loaders["val"])
            #print(len(current_loaders["val"]))
            #self.trainer.fit(model, current_loaders["val"], current_loaders["val"])

            done_keys = (None,None)
            for iteration_num,(starting_key_id,current_key_id) in enumerate(iteration_ranges):
                print("CURRENT PERIOD:", list(all_train_data.keys())[starting_key_id], "(incl.) - ", list(all_train_data.keys())[current_key_id-1],"(incl.)")
                
                if (starting_key_id,current_key_id) != done_keys:
                    print("GETTING NEW DATA")
                    new_data = data_utils.get_new_data_in_range(all_train_data, starting_key_id, current_key_id)
                else:
                    print("DATA ALREADY ACQUIRED")
                    new_data = rem_data
                    if new_data is None:
                        print("!!!"*10,"NO REMAINING DATA","!!!"*10)
                        break
                print("NEW TRAINING DATA")
                keep_all_new = doing_warm_start and iteration_num==0

                print("MERGE NEW DATA")
                current_data, rem_data, (new_positives, new_negatives) = AL.merge_new_data(current_data, new_data,
                                                                                            self.cfg, keep_all_new, #FEDE_WHAT_TO_DO
                                                                                            model, self.trainer)

                print("CHANGE CURRENT DATALOADER")
                current_loaders["train"] = DataLoader(current_data["train"],batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.workers_available, pin_memory=True)

                print("FINISH GETTING NEW TRAINING DATA")
                print("data['train'].shape, ", len(current_data['train']))
                print("data['val'].shape, ", len(current_data['val']))

                #save new_positives/negatives and current_positives/negatives
                current_positives, current_negatives = data_utils.compute_new_positives_negatives(model, current_loaders["train"])
                
                #LOG POSS/NEGS; MOVE TO DATA UTILS?
                #wandb.log("new_positives", new_positives)
                #wandb.log("new_negatives", new_negatives)
                #wandb.log("current_positives", current_positives)
                #wandb.log("current_negatives", current_negatives)

                #all_true_false_nums[sample,iteration_num,:] += np.array([new_positives, new_negatives,
                #                                                        current_positives,current_negatives])
                
                if rem_data is not None:
                    print("REM SHAPES",len(rem_data))#, len(new_y))

                if self.cfg.retrain_from_scratch: #Re-initialize model
                    model = graph_model.initialize_graph_model(self.cfg)

                print("TRAINING MODEL")
                self.trainer.fit(model, current_loaders["train"], current_loaders["val"])

                print("COMPUTE TEST METRICS")
                #rs = data_utils.model_evaluate_per_month(model, all_test_data)
            
                self.trainer.test(model, current_loaders["test"], ckpt_path="best")

                #wandb.finish()
                
                #all_rs[sample,iteration_num,:,:] += rs

                done_keys = starting_key_id,current_key_id
        
        #with open(results_filename, 'wb') as f:
        #    np.save(f, all_rs)

        #with open(all_pos_neg_filename, 'wb') as f:
        #    np.save(f, all_true_false_nums)
        
    def end_pipeline(self):
        wandb.finish()
        