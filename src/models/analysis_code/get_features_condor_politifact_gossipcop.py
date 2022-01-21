import pandas as pd
import numpy as np
import os
import random
import pickle as pkl
import json
import time
from datetime import datetime
from dateutil import parser

# pd.set_option("display.max_rows", None, "display.max_columns", None)

# full_dataset_url = "../../condor_test/data/raw/full_dataset_condor_gossipcop_politifact.csv"

# condor_urls = "../../condor_test/data/raw/ground_truth_diffusion_data_condor/"
# gossipcop_urls = "../../condor_test/data/raw/ground_truth_diffusion_data_gossipcop/"
# politifact_urls = "../../condor_test/data/raw/ground_truth_diffusion_data_politifact/"
# twitter_user_data = "../../condor_test/data/raw/all_twitter_accounts/user_profiles/"

# df = pd.read_csv("../../condor_test/data/raw/full_dataset_condor_gossipcop_politifact.csv")
# # We drop URLs that for whatever reasons where labeled as to discard
# df_filtered = df[df['discard (1) - doubt (2) - to discuss'] != 1]
# # df_filtered = df_filtered[df_filtered['timestamp_first_tweet'] >= 1451714232]
# # df_filtered = df_filtered[df_filtered['timestamp_first_tweet'] <= 1578884956]
# # We drop duplicate news
# df_filtered.sort_values(['duplicate_cluster', 'total_tweets_after_480_hours'], ascending=False)
# df_no_duplicates = df_filtered.drop_duplicates(subset=['duplicate_cluster'], keep='last')

# already_done = os.listdir('../../condor_test/data/processed/url_to_user_sequence/')

# for index, row in df_no_duplicates.iterrows():
#     id = row["id"]
#     if '{}.pickle'.format(id) in already_done: continue
#     if row['dataset'] == 'condor':
#         user_posting_user_sequence = os.path.join(condor_urls, id+'.pickle')
#         pickle_in = open(user_posting_user_sequence,"rb")
#         tweet_sequence = pkl.load(pickle_in)
#     elif row['dataset'] == 'gossip':
#         if row["tpfc_rating_encoding"] == 0:
#             user_posting_user_sequence = os.path.join(gossipcop_urls, 'gossipcop_real', id, 'tweets.json')
#         else:
#             user_posting_user_sequence = os.path.join(gossipcop_urls, 'gossipcop_fake', id, 'tweets.json')
        
#         user_posting_user_sequence = open(user_posting_user_sequence)
#         tweet_sequence = json.load(user_posting_user_sequence)
#     elif row['dataset'] == 'poli':
#         if row["tpfc_rating_encoding"] == 0:
#             user_posting_user_sequence = os.path.join(politifact_urls, 'politifact_real', id, 'tweets.json')
#         else:
#             user_posting_user_sequence = os.path.join(politifact_urls, 'politifact_fake', id, 'tweets.json')
#         user_posting_user_sequence = open(user_posting_user_sequence)
#         tweet_sequence = json.load(user_posting_user_sequence)

#     if tweet_sequence: 
#         key = list(tweet_sequence.keys())[0]

#         user_user_sequence = []

#         for tweet in tweet_sequence[key]:
#             if 'author_id' in tweet.keys():
#                 author_id = tweet['author_id']
#                 timestamp = int(datetime.timestamp(parser.parse(tweet['created_at'])))
#             else:
#                 author_id = tweet['user_id']
#                 timestamp = tweet['created_at']
#             user_user_sequence.append((str(author_id), timestamp))
        
#         with open('../../condor_test/data/processed/url_to_user_sequence/'+id+'.pickle', "wb") as fp:
#             pkl.dump(user_user_sequence, fp)

# >>> with open("test.txt", "rb") as fp:   # Unpickling
# ...   b = pickle.load(fp)

#already_done = os.listdir('../../condor_test/data/processed/url_to_user_sequence/')

# for url in already_done:
#     print('../../condor_test/data/processed/url_to_user_sequence/'+url)
#     with open('../../condor_test/data/processed/url_to_user_sequence/'+url, "rb") as fp: 
#         b = pkl.load(fp)
#     print(len(b))

# users_dir = '../../condor_test/data/raw/all_twitter_accounts/user_profiles/'

# users = os.listdir(users_dir)

# all_users_char = {}

# for user in users:
#     user_file = os.path.join(users_dir, user)
#     user_posting_user_sequence = open(user_file)
#     user_profile = json.load(user_posting_user_sequence)
#     id = user_profile['id']
#     all_users_char[id] = []
    
#     all_users_char[id].append(len(user_profile['name']))
#     all_users_char[id].append(len(user_profile['username']))
#     all_users_char[id].append(len(user_profile['description']))

#     all_users_char[id].append(user_profile['public_metrics']['followers_count'])
#     all_users_char[id].append(user_profile['public_metrics']['following_count'])
#     all_users_char[id].append(user_profile['public_metrics']['tweet_count'])
#     all_users_char[id].append(user_profile['public_metrics']['listed_count'])

#     current_time = datetime.now()
#     ts_now = int(datetime.timestamp(current_time))
#     ts_user = int(datetime.timestamp(parser.parse(user_profile['created_at'])))
#     all_users_char[id].append(int((ts_now - ts_user) /(3600*24)))

#     all_users_char[id].append(int('url' in user_profile))
#     all_users_char[id].append(int('location' in user_profile))
#     all_users_char[id].append(int('profile_image_url' in user_profile))
#     all_users_char[id].append(int(user_profile['protected']))
#     all_users_char[id].append(int(user_profile['verified']))

#     print(all_users_char[id])

df = pd.read_csv("../../../data/final_dataset_condor_gossipcop_politifact.csv")
# We drop URLs that for whatever reasons where labeled as to discard
#df_filtered = df[df['timestamp_first_tweet'] > 1483228799]
#df_filtered = df_filtered[df_filtered['timestamp_first_tweet'] < 1577836801

df = df[df['dataset'] == 'poli']

with open('../../../../condor_test/data/processed/all_users_features.pickle', "rb") as fp:
    all_users = pkl.load(fp)

already_done = os.listdir('../../../../condor_test/data/processed/url_to_user_sequence/')

with open("../../../data/poli_evaluation_dataset_ids.txt", "rb") as fp:
   evaluation_dataset_ids = pkl.load(fp)

# months = df['month_year'].unique()

# for month in months:
#     #if '2017' not in month and '2018' not in month and '2019' not in month: continue

#     print(month)

#     monthly_df = df[df['month_year'] == month]
    
#     y = []
#     x = []

#     for index, row in monthly_df.iterrows():
#         url_id = row["id"]
#         if '{}.pickle'.format(url_id) not in already_done: continue
#         if url_id not in evaluation_dataset_ids: continue
#         with open('../../../../condor_test/data/processed/url_to_user_sequence/{}.pickle'.format(url_id), "rb") as fp: 
#             url = pkl.load(fp)
#         y.append(int(monthly_df[monthly_df['id']==url_id].iloc[0]['tpfc_rating_encoding']))
#         i = 0
#         url_features = []
#         for spreading_user in url:
#             user_id = spreading_user[0]
#             if i < 100 and user_id in all_users.keys():
#                 url_features.append(all_users[user_id])
#                 i += 1
#         while i < 100:
#             url_features.append([0]*13)
#             i += 1
#         x.append(url_features)

#     x = np.array(x)
#     y = np.array(y)

#     print('x shape', x.shape)
#     print('y shape', y.shape)

#     np.save('../../../data/features/condor/eval_{}_x.npy'.format(month), x)
#     np.save('../../../data/features/condor/eval_{}_y.npy'.format(month), y)



y = []
x = []

j = 0 

for index, row in df.iterrows():
    url_id = row["id"]
    if '{}.pickle'.format(url_id) not in already_done: continue
    if url_id not in evaluation_dataset_ids: continue
    evaluation_dataset_ids.append(url_id)
    j+=1
    print(j)
    with open('../../../../condor_test/data/processed/url_to_user_sequence/{}.pickle'.format(url_id), "rb") as fp: 
        url = pkl.load(fp)
    y.append(int(df[df['id']==url_id].iloc[0]['tpfc_rating_encoding']))
    i = 0
    url_features = []
    for spreading_user in url:
        user_id = spreading_user[0]
        if i < 100 and user_id in all_users.keys():
            url_features.append(all_users[user_id])
            i += 1
    while i < 100:
        url_features.append([0]*13)
        i += 1
    x.append(url_features)

x = np.array(x)
y = np.array(y)

print('x shape', x.shape)
print('y shape', y.shape)

np.save('../../../data/features/politifact/evaluation_x.npy', x)
np.save('../../../data/features/politifact/evaluation_y.npy', y)

