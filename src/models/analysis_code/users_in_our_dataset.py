import pandas as pd
import numpy as np
import os
import random
import pickle as pkl
import json
import time
from datetime import datetime
from dateutil import parser

# df = pd.read_csv("../../../data/final_dataset_condor_gossipcop_politifact.csv")

# users_list = []

# already_done = os.listdir('../../../../condor_test/data/processed/url_to_user_sequence/')

# for index, row in df.iterrows():
#     id = row["id"]
#     if id+'.pickle' not in already_done: continue
#     with open('../../../../condor_test/data/processed/url_to_user_sequence/'+id+'.pickle', "rb") as fp: 
#         twitter_user_seq = pkl.load(fp)

#     i = 0

#     for user in twitter_user_seq:
#         users_list.append(int(user[0]))
#         i += 1
#         if i == 20: break

# print(len(users_list))
# print(len(list(set(users_list))))

# users_list = list(set(users_list))

# open_file = open('../../../../condor_test/data/raw/user_list_first_20_users.pickle', "wb")
# pkl.dump(users_list, open_file)
# open_file.close()


# --------------------------------------------------------------------------------

# with open('../../../../condor_test/data/raw/user_list_first_20_users.pickle', 'rb') as handle:
#     user_list = pkl.load(handle) 

# missing_followees = 0
# missing_liked_tweets = 0

# user_liked_tweets = os.listdir('../../../../condor_test/data/raw/all_twitter_accounts/user_liked_tweets/')
# user_followees = os.listdir('../../../../condor_test/data/raw/all_twitter_accounts/user_followees/')

# j = 0 

# for user in user_list:

#     j += 1
#     print(j)

#     if str(user)+'.json' not in user_liked_tweets:
#         missing_followees += 1

#     if str(user)+'.json' not in user_followees:
#         missing_liked_tweets += 1

# print('missing_followees', missing_followees)
# print('missing_liked_tweets', missing_liked_tweets)

# --------------------------------------------------------------------------------

df = pd.read_csv("../../../data/final_dataset_condor_gossipcop_politifact.csv")

condor_urls = "../../../../condor_test/data/raw/ground_truth_diffusion_data_condor/"
gossipcop_urls = "../../../../condor_test/data/raw/ground_truth_diffusion_data_gossipcop/"
politifact_urls = "../../../../condor_test/data/raw/ground_truth_diffusion_data_politifact/"

for index, row in df.iterrows():
    id = row["id"]
    if row['dataset'] == 'condor':
        user_posting_user_sequence = os.path.join(condor_urls, id+'.pickle')
        pickle_in = open(user_posting_user_sequence,"rb")
        tweet_sequence = pkl.load(pickle_in)
    elif row['dataset'] == 'gossip':
        if row["tpfc_rating_encoding"] == 0:
            user_posting_user_sequence = os.path.join(gossipcop_urls, 'gossipcop_real', id, 'tweets.json')
        else:
            user_posting_user_sequence = os.path.join(gossipcop_urls, 'gossipcop_fake', id, 'tweets.json')
        
        user_posting_user_sequence = open(user_posting_user_sequence)
        tweet_sequence = json.load(user_posting_user_sequence)
    elif row['dataset'] == 'poli':
        if row["tpfc_rating_encoding"] == 0:
            user_posting_user_sequence = os.path.join(politifact_urls, 'politifact_real', id, 'tweets.json')
        else:
            user_posting_user_sequence = os.path.join(politifact_urls, 'politifact_fake', id, 'tweets.json')
        user_posting_user_sequence = open(user_posting_user_sequence)
        tweet_sequence = json.load(user_posting_user_sequence)

    for i in range(10):
        try:
            print(tweet_sequence['tweets'][i].keys())
        except:
            print("less")