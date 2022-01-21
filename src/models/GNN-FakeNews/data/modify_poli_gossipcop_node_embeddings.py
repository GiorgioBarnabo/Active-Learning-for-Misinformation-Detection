import pickle
import pandas as pd
import time
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from statistics import mean

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

df = pd.read_csv("../../../../data/final_dataset_condor_gossipcop_politifact.csv")

url_ids_list = df["id"].tolist()

#df[df['id'] == ''].iloc[0]['share_title']

twitter_users_dir = '../../../../../condor_test/data/raw/all_twitter_accounts/'

# twitter_accounts = []

# with open('pol_id_twitter_mapping.pkl', 'rb') as handle:
#     dataset_users = pickle.load(handle)
# for key in dataset_users:
#     if 'politifact' not in dataset_users[key]:
#         twitter_accounts.append(dataset_users[key])

# with open('gos_id_twitter_mapping.pkl', 'rb') as handle:
#     dataset_users = pickle.load(handle)
# for key in dataset_users:
#     if 'gossipcop' not in dataset_users[key]:
#         twitter_accounts.append(dataset_users[key])

# print(len(twitter_accounts))

# twitter_accounts = list(set(twitter_accounts))

# print(len(twitter_accounts))

# missing = 0

# for account in twitter_accounts:
#     if not os.path.isfile(os.path.join(twitter_users_dir, 'user_profiles', account+'.json')):
#         missing += 1

# print(missing)


# with open('pol_id_twitter_mapping.pkl', 'rb') as handle:
#     dataset_users = pickle.load(handle)

# with open('gos_id_twitter_mapping.pkl', 'rb') as handle:
#     dataset_users = pickle.load(handle)

with open('condor_id_twitter_mapping.pkl', 'rb') as handle:
    dataset_users = pickle.load(handle)

profile_feature = []
bert_feature = []

# mean_profile = []
# mean_bert = []

# for i in range(len(dataset_users)):
#     if dataset_users[i] not in url_ids_list and os.path.isfile(os.path.join(twitter_users_dir, 'user_profiles', dataset_users[i]+'.json')):
#         with open(os.path.join(twitter_users_dir, 'user_profiles', dataset_users[i]+'.json')) as json_file:
#             data = json.load(json_file)
#             mean_profile.append(data['user_features'])
#         mean_bert.append(list(np.load(os.path.join(twitter_users_dir, 'all_twitter_account_description_embeddings', dataset_users[i]+'.npy'))))


# mean_profile = np.mean(np.array(mean_profile), axis=0).tolist()
# mean_bert = np.mean(np.array(mean_bert), axis=0).tolist()

for i in range(len(dataset_users)):
    if dataset_users[i] in url_ids_list:
        print('NEWS!!')
        profile_feature.append(np.random.rand(13))
        try:
            bert_feature.append(model.encode(df[df['id'] == dataset_users[i]].iloc[0]['share_title']).tolist())
        except:
            bert_feature.append(np.random.rand(768))
    elif not os.path.isfile(os.path.join(twitter_users_dir, 'user_profiles', dataset_users[i]+'.json')):
        #print('MISSING PROFILE!!')
        profile_feature.append(np.random.rand(13))
        bert_feature.append(np.random.rand(768))
    else:
        #print('NOT MISSING PROFILE!!')
        with open(os.path.join(twitter_users_dir, 'user_profiles', dataset_users[i]+'.json')) as json_file:
            data = json.load(json_file)
            profile_feature.append(data['user_features'])
            bert_feature.append(np.load(os.path.join(twitter_users_dir, 'all_twitter_account_description_embeddings', dataset_users[i]+'.npy')).tolist())

profile_feature = np.array(profile_feature)
bert_feature = np.array(bert_feature)

print(profile_feature.shape)
print(bert_feature.shape)

np.savez('condor/raw/new_profile_feature.npz', profile_feature)
np.savez('condor/raw/new_bert_feature.npz', bert_feature)

## --------------------------------------------------------------

# gossipcop_bert = np.load('gossipcop/raw/new_bert_feature.npz')
# politifact_bert = np.load('politifact/raw/new_bert_feature.npz')

# gossipcop_profile = np.load('gossipcop/raw/new_profile_feature.npz')
# politifact_profile = np.load('politifact/raw/new_profile_feature.npz')

# print(gossipcop_bert['arr_0'])
# print(politifact_bert['arr_0'])
# print(gossipcop_profile['arr_0'])
# print(politifact_profile['arr_0'])

#downloaded_users_list = os.listdir('../data/raw/all_twitter_accounts/user_profiles/')