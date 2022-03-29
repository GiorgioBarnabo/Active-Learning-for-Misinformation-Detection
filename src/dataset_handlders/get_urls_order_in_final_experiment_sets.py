import pickle 
import os

os.chdir(
    "/home/barnabog/Online-Active-Learning-for-Misinformation-Detection/data/"  #ATTENTO_FEDE
)

with open("accessory_files/condor_id_twitter_mapping.pkl", 'rb') as f:
    results = pickle.load(f)

print(results[0])

ordered_list_of_urls = []

print('c98h165lbnqhbkx'.isnumeric())


for key in results.keys():
    str_to_check = str(results[key])
    if not str_to_check.isnumeric():
        print("ok")
        ordered_list_of_urls.append(results[key])

print(ordered_list_of_urls)

with open("accessory_files/condor_ordered_list_used_in_experiments.pkl", 'wb') as f:
    pickle.dump(ordered_list_of_urls, f)