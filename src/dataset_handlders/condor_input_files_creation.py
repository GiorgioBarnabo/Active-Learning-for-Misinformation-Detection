import pickle
import pandas as pd
import time
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from statistics import mean
import random
import csv

# save new files in "condor/raw"

df = pd.read_csv("../../../../data/final_dataset_condor_gossipcop_politifact.csv")

A = []
graph_labels = []
node_graph_id = []
test_idx = []
train_idx = []
val_idx = []
condor_id_twitter_mapping_temp = {}
condor_id_twitter_mapping = {}
node_index = 0
graph_index = 0

stop = 0

for i in range(len(df)):
    if df.loc[i, "dataset"] == "condor" and df.loc[i, "id"] not in ['8ujoxz2a8rbefx8']:
        print(i)
        id = df.loc[i, "id"]
        graph_labels.append(df.loc[i, "tpfc_rating_encoding"])
        condor_id_twitter_mapping_temp[node_index] = id
        root = node_index
        node_graph_id.append(graph_index)
        node_index += 1
        tweets_seq_url = '../../../../../condor_test/data/raw/ground_truth_diffusion_data_condor/{}.pickle'.format(id)
        with open(tweets_seq_url, 'rb') as handle:
            tweets_seq = pickle.load(handle)[id][0:1000]
        tweet_id_to_author_id = {}
        for tweet in tweets_seq:
            if tweet['author_id'] not in condor_id_twitter_mapping_temp.values():
                condor_id_twitter_mapping_temp[node_index] = tweet['author_id']
                node_graph_id.append(graph_index)
                node_index += 1
                tweet_id_to_author_id[tweet['id']] = tweet['author_id']
        key_list = list(condor_id_twitter_mapping_temp.keys())
        val_list = list(condor_id_twitter_mapping_temp.values())
        for tweet in tweets_seq:
            position = val_list.index(tweet['author_id'])
            if 'referenced_tweets' not in tweet.keys():
                A.append([root, key_list[position]])
                #node_graph_id.append(graph_index)
            else:
                ref_id = str(tweet['referenced_tweets'][0]['id'])
                if ref_id in tweet_id_to_author_id.keys():
                    print("hurray!!!!!!!!!!!!!!")
                    position_origin = val_list.index(tweet_id_to_author_id[ref_id])
                    A.append([key_list[position_origin], key_list[position]])
                    #node_graph_id.append(graph_index)
        rand = random.random()
        if rand <= 0.2:
            train_idx.append(graph_index)
        elif rand <= 0.3:
            val_idx.append(graph_index)
        else:
            test_idx.append(graph_index)
        graph_index += 1

        condor_id_twitter_mapping.update(condor_id_twitter_mapping_temp)
        condor_id_twitter_mapping_temp = {}
    #     stop += 1
    # if stop > 3:
    #     break
    

# "condor/raw"

np.save('condor/raw/graph_labels.npy', np.array(graph_labels))
np.save('condor/raw/node_graph_id.npy', np.array(node_graph_id))
np.save('condor/raw/test_idx.npy', np.array(test_idx))
np.save('condor/raw/val_idx.npy', np.array(val_idx))
np.save('condor/raw/train_idx.npy', np.array(train_idx))
with open('condor/raw/A.txt', 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
    csv_writer.writerows(A)
with open('condor_id_twitter_mapping.pkl', 'wb') as handle:
    pickle.dump(condor_id_twitter_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)


#twitter_users_dir = '../../../../../condor_test/data/raw/all_twitter_accounts/'

# all_twitter_account_description_embeddings (npy):

# list(np.load(os.path.join(twitter_users_dir, 'all_twitter_account_description_embeddings', dataset_users[i]+'.npy'))

# user_profiles (json): user_features

# with open(os.path.join(twitter_users_dir, 'user_profiles', dataset_users[i]+'.json')) as json_file:
#     data = json.load(json_file)

# graph_labels = np.load('data/politifact/raw/graph_labels.npy')
# node_graph_id = np.load('data/politifact/raw/node_graph_id.npy')
# test_idx = np.load('data/politifact/raw/test_idx.npy')

# print(node_graph_id.shape)


