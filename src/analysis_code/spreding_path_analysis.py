import pandas as pd
import numpy as np
import os
import pickle as pkl
import json
import time
from datetime import datetime
from dateutil import parser

full_dataset_url = "../../condor_test/data/raw/full_dataset_condor_gossipcop_politifact.csv"
df = pd.read_csv("../../condor_test/data/raw/full_dataset_condor_gossipcop_politifact.csv")
# We drop URLs that for whatever reasons where labeled as to discard
df_filtered = df[df['discard (1) - doubt (2) - to discuss'] != 1]
#df_filtered = df_filtered[df_filtered['timestamp_first_tweet'] >= 1451714232]
#df_filtered = df_filtered[df_filtered['timestamp_first_tweet'] <= 1578884956]
# We drop duplicate news
df_filtered.sort_values(['duplicate_cluster', 'total_tweets_after_480_hours'], ascending=False)
df_no_duplicates = df_filtered.drop_duplicates(subset=['duplicate_cluster'], keep='last')

df_no_duplicates = df_no_duplicates.sort_values(['timestamp_first_tweet'], inplace=False)

df_no_duplicates.drop([
    'discard (1) - doubt (2) - to discuss',
    'notes',
    'average_tweet_increat_between_1_2_hours',
    'average_tweet_increat_between_2_5_hours',
    'average_tweet_increat_between_5_10_hours',
    'average_tweet_increat_between_10_15_hours',
    'average_tweet_increat_between_15_24_hours',
    'average_tweet_increat_between_24_50_hours',
    'average_tweet_increat_between_50_120_hours',
    'average_tweet_increat_between_120_480_hours'], axis=1, inplace=True)

df_no_duplicates['month_year'] = pd.to_datetime(df_no_duplicates['date_first_tweet']).dt.to_period('M')

df_no_duplicates.to_csv("../../condor_test/data/raw/final_dataset_condor_gossipcop_politifact.csv")

with open('../../condor_test/data/processed/all_users_features.pickle', "rb") as fp:
    all_users = pkl.load(fp)

already_done = os.listdir('../../condor_test/data/processed/url_to_user_sequence/')

monthly_sequences = {}

k = 0

for index, row in df_no_duplicates.iterrows():

    k += 1
    print(k)

    url_id = row["id"]
    if '{}.pickle'.format(url_id) not in already_done: continue

    tpfc_rating = int(df_no_duplicates[df_no_duplicates['id']==url_id].iloc[0]['tpfc_rating_encoding'])
    month = str(df_no_duplicates[df_no_duplicates['id']==url_id].iloc[0]['month_year'])

    monthly_sequences[month] = monthly_sequences.get(month, {})
    monthly_sequences[month][tpfc_rating] = monthly_sequences[month].get(tpfc_rating, {})
    
    with open('../../condor_test/data/processed/url_to_user_sequence/{}.pickle'.format(url_id), "rb") as fp: 
        url = pkl.load(fp)
    
    i = 0
    for spreading_user in url:
        user_id = spreading_user[0]
        if i < 100 and user_id in all_users.keys():
            for j in range(len(all_users[user_id])):
                monthly_sequences[month][tpfc_rating][j] = monthly_sequences[month][tpfc_rating].get(j, np.zeros((2, 100)))
                monthly_sequences[month][tpfc_rating][j][0][i] += all_users[user_id][j]
                monthly_sequences[month][tpfc_rating][j][1][i] += 1
        i += 1
        if i > 99: break


a_file = open("../../condor_test/data/processed/monthly_buckets_statistics.pickle", "wb")
pkl.dump(monthly_sequences, a_file)
a_file.close()

a_file = open("../../condor_test/data/processed/monthly_buckets_statistics.pickle", "rb")
monthly_sequences = pkl.load(a_file)

tpfc_labels = {0: 'true', 1: 'false'}

featues = {
    0: 'len_name',
    1: 'len_username', 
    2: 'len_description', 
    3: 'num_followers',
    4: 'num_followees',
    5: 'total_tweets',
    6: 'lists_count',
    7: 'profile_age',
    8: 'bool_url',
    9: 'bool_location',
    10: 'bool_profile_image',
    11: 'bool_protected',
    12: 'bool_verified'}

to_plot_values = {}
to_plot_freq = {}


for month in monthly_sequences.keys():
    if '2016' in month or '2017' in month or '2018' in month or '2019' in month:
        for tpfc in monthly_sequences[month].keys():
            for feature in monthly_sequences[month][tpfc].keys():
                name = month + ' | ' + tpfc_labels[tpfc]  + ' | ' + featues[feature]
                to_plot_values[name] = monthly_sequences[month][tpfc][feature][0, :] / monthly_sequences[month][tpfc][feature][1, :]
                to_plot_freq[name] = monthly_sequences[month][tpfc][feature][1, :]
                monthly_sequences[month][tpfc][feature]

a_file = open("../../condor_test/data/processed/monthly_buckets_feaures_evolution_values.pickle", "wb")
pkl.dump(to_plot_values, a_file)
a_file.close()

a_file = open("../../condor_test/data/processed/monthly_buckets_feaures_evolution_freq.pickle", "wb")
pkl.dump(to_plot_freq, a_file)
a_file.close()

df1 = pd.DataFrame(to_plot_values)
df2 = pd.DataFrame(to_plot_freq)

df1.to_csv("../../condor_test/data/processed/monthly_buckets_feaures_evolution_values.csv")
df2.to_csv("../../condor_test/data/processed/monthly_buckets_feaures_evolution_freq.csv")




