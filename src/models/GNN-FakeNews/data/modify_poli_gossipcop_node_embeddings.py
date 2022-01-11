import pickle
import pandas as pd
import time
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from statistics import mean

'''
#model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

df = pd.read_csv("../../../../data/final_dataset_condor_gossipcop_politifact.csv")

#df[df['id'] == ''].iloc[0]['share_title']

twitter_users_dir = '../../../../../condor_test/data/raw/all_twitter_accounts/'

# with open('pol_id_twitter_mapping.pkl', 'rb') as handle:
#     dataset_users = pickle.load(handle)

with open('gos_id_twitter_mapping.pkl', 'rb') as handle:
    dataset_users = pickle.load(handle)

profile_feature = []
bert_feature = []

mean_profile = []
mean_bert = []

for i in range(len(dataset_users)):
    if 'gossipcop' not in dataset_users[i] and os.path.isfile(os.path.join(twitter_users_dir, 'user_profiles', dataset_users[i]+'.json')):
        with open(os.path.join(twitter_users_dir, 'user_profiles', dataset_users[i]+'.json')) as json_file:
            data = json.load(json_file)
            mean_profile.append(data['user_features'])
        mean_bert.append(list(np.load(os.path.join(twitter_users_dir, 'all_twitter_account_description_embeddings', dataset_users[i]+'.npy'))))

mean_profile = np.mean(np.array(mean_profile), axis=0).tolist()
mean_bert = np.mean(np.array(mean_bert), axis=0).tolist()

for i in range(len(dataset_users)):
    if 'gossipcop' in dataset_users[i] or not os.path.isfile(os.path.join(twitter_users_dir, 'user_profiles', dataset_users[i]+'.json')):
        profile_feature.append(mean_profile)
        bert_feature.append(mean_bert)
    else:
        with open(os.path.join(twitter_users_dir, 'user_profiles', dataset_users[i]+'.json')) as json_file:
            data = json.load(json_file)
            profile_feature.append(data['user_features'])
            bert_feature.append(np.load(os.path.join(twitter_users_dir, 'all_twitter_account_description_embeddings', dataset_users[i]+'.npy')).tolist())

profile_feature = np.array(profile_feature)
bert_feature = np.array(bert_feature)

print(profile_feature.shape)
print(bert_feature.shape)

np.savez('gossipcop/raw/new_profile_feature_new.npz', profile_feature)
np.savez('gossipcop/raw/new_bert_feature_new.npz', bert_feature)

'''

gossipcop_bert = np.load('gossipcop/raw/new_bert_feature.npz')
politifact_bert = np.load('politifact/raw/new_bert_feature.npz')

gossipcop_profile = np.load('gossipcop/raw/new_profile_feature.npz')
politifact_profile = np.load('politifact/raw/new_profile_feature.npz')

print(gossipcop_bert['arr_0'].shape)
print(politifact_bert['arr_0'].shape)
print(gossipcop_profile['arr_0'].shape)
print(politifact_profile['arr_0'].shape)

#downloaded_users_list = os.listdir('../data/raw/all_twitter_accounts/user_profiles/')