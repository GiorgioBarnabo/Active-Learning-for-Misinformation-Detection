import sys, os, argparse
sys.path.append('..')

project_folder = os.path.join('../'*5)

import numpy as np

#import our scripts
from . import AL
from . import data_utils
from . import graph_model

#from omegaconf import OmegaConf,open_dict


class Pipeline():
    def __init__(self, cfg, experiment_id):
        self.cfg = cfg
        self.experiment_id = experiment_id

    def prepare_graph_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=self.cfg.data_params.batch_size, help='batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
        parser.add_argument('--nhid', type=int, default=128, help='hidden size')
        parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
        parser.add_argument('--epochs', type=int, default=self.cfg.model_params.epochs, help='maximum number of epochs')
        parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
        parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
        parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
        parser.add_argument('--feature', type=str, default=self.cfg.model_params.graph_feature, help='feature type, [profile, spacy, bert, content]')
        parser.add_argument('--model', type=str, default=self.cfg.model_params.graph_model_name, help='model type, [gcn, gat, sage]')
        return parser.parse_args()


    def prepare_pipeline(self):
        #with open_dict(self.cfg): #needed if trying to set attributes not in config
        self.graph_args = self.prepare_graph_parser()

        #Change "inf" to np.inf
        for i,x in enumerate(self.cfg.data_params.warm_start_years):
            if x=="inf":
                self.cfg.data_params.warm_start_years[i] = np.inf

        if self.cfg.AL_params.train_last_samples=="inf":
            self.cfg.AL_params.train_last_samples = np.inf

        if self.cfg.AL_params.val_last_samples=="inf":
            self.cfg.AL_params.val_last_samples = np.inf

        ####Compute num_urls_k_list:
        if "num_urls_k" not in self.cfg.AL_params:
            if self.cfg.AL_params.offline_AL>0:
                self.cfg.AL_params.num_urls_k = int(self.cfg.AL_params.tot_num_checked_urls)
            else:
                print("TO BE DEFINED ---> divide tot_num_... for number of time-steps?", error)

        #Set results folder
        self.results_folder = os.path.join(project_folder, 'out', "results",str(self.experiment_id))#, self.cfg.experiment_params.results_set, self.cfg.data_params.dataname)
        #create results_folder not exists
        if not os.path.isdir(self.results_folder):
            os.makedirs(self.results_folder)

        self.data_folder = os.path.join(project_folder, 'data', 'graph', self.cfg.data_params.dataname)


    def run_pipeline(self):
        all_train_data, all_val_data, all_test_data = data_utils.load_graph_data(self.data_folder) #to load all data
        print("all_train_data.keys", all_train_data.keys())
        print("all_val_data.keys", all_val_data.keys())
        print("all_test_data.keys", all_test_data.keys())
        
        first_key = list(all_train_data.keys())[0]
        self.graph_args.num_features = all_train_data[first_key][0].num_features
        self.graph_args.num_classes = 2 #all_train_data[first_key].num_classes

        #WARM-START PREPARATION
        warm_start_starting_key_id,warm_start_ending_key_id = data_utils.get_warm_start_key_ids(all_train_data.keys(), self.cfg.data_params.warm_start_years)
        print(warm_start_starting_key_id,warm_start_ending_key_id)
        
        doing_warm_start = True
        if warm_start_starting_key_id==len(all_train_data)-1 or warm_start_ending_key_id==0:
            print("!!! NOT DOING WARM START !!!")
            warm_start_starting_key_id = 0
            if self.cfg.AL_params.offline_AL>0:
                warm_start_ending_key_id = len(all_train_data)
            else:
                warm_start_ending_key_id = 1
            doing_warm_start = False

        #PREPARE ITERATION RANGES
        if self.cfg.AL_params.offline_AL>0:
            if doing_warm_start:
                iteration_ranges = list(zip([warm_start_starting_key_id]+[warm_start_ending_key_id]*self.cfg.AL_params.offline_AL,
                                            [warm_start_ending_key_id]+[len(all_train_data)]*self.cfg.AL_params.offline_AL))
            else:
                iteration_ranges = list(zip([warm_start_starting_key_id]*self.cfg.AL_params.offline_AL,
                                            [len(all_train_data)]*self.cfg.AL_params.offline_AL))
        else:
            iteration_ranges = list(zip([warm_start_starting_key_id]+list(range(warm_start_ending_key_id, len(all_train_data))),
                                    range(warm_start_ending_key_id, len(all_train_data)+1)))
        print("IT_RNG:",iteration_ranges)
            
        results_filename = os.path.join(self.results_folder,"rs.npy")#os.path.join(results_folder, '_ALm_' + AL_method + '_k_' + str(num_urls_k) + '.npy')
        all_pos_neg_filename = os.path.join(self.results_folder,"pos_neg.npy")
        if self.cfg.AL_params.offline_AL>0:
            all_rs = np.zeros((self.cfg.experiment_params.nb_samples,
                            self.cfg.AL_params.offline_AL + 1*doing_warm_start,
                            len(all_test_data),
                            8,))
            all_true_false_nums = np.zeros((self.cfg.experiment_params.nb_samples,
                                            self.cfg.AL_params.offline_AL + 1*doing_warm_start,
                                            4,))
        
        else:
            print("ONLINE AL TO BE DEFINED",error)
        
        for sample in range(self.cfg.experiment_params.nb_samples):
            current_data = {"train": None,
                            "val": all_val_data}

            #Initialize model
            model = graph_model.initialize_graph_model(self.graph_args, self.cfg.experiment_params.starting_seed)

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

                current_data, rem_data, (new_positives, new_negatives) = AL.merge_new_data(current_data, new_data,
                                                                                            self.cfg.AL_params, keep_all_new,
                                                                                            model)
                print("FINISH GETTING NEW TRAINING DATA")
                print("data['train'].shape, ", len(current_data['train']))
                print("data['val'].shape, ", len(current_data['val']))

                #save new_positives/negatives and current_positives/negatives
                current_positives, current_negatives = data_utils.compute_new_positives_negatives(model, current_data)
                all_true_false_nums[sample,iteration_num,:] += np.array([new_positives, new_negatives,
                                                                        current_positives,current_negatives])

                
                if rem_data is not None:
                    print("REM SHAPES",len(rem_data))#, len(new_y))

                if self.cfg.AL_params.retrain_from_scratch: #Re-initialize model
                    model = graph_model.initialize_graph_model(self.graph_args, self.cfg.experiment_params.starting_seed)

                print("TRAINING MODEL")
                graph_model.train_graph_model(model, self.graph_args, current_data["train"])
                
                print("COMPUTE TEST METRICS")
                rs = data_utils.model_evaluate_per_month(model, all_test_data)
            
                all_rs[sample,iteration_num,:,:] += rs

                done_keys = starting_key_id,current_key_id
        
        with open(results_filename, 'wb') as f:
            np.save(f, all_rs)

        with open(all_pos_neg_filename, 'wb') as f:
            np.save(f, all_true_false_nums)
        
