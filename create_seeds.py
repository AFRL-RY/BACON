# run_create_seeds.py

import json

import random

import shutil
import os

if __name__ == "__main__":


    hyperparameter_file = 'hyperparameters.json'
    datasets_folder = './datasets'
    save_folder = './output/'
    seeds_file = save_folder + 'seeds.txt'
    file_list = [datasets_folder,save_folder]


    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    seed_count_total = jsonObject['seed_count_total']

    #Ensure files have a home
    for file in file_list:
        if os.path.exists(file) == False:
            os.mkdir(file)

    # Generate random set of seeds
    num_low = 0
    num_high = 10000
    set_of_seeds = set()
    while len(set_of_seeds) < seed_count_total:
        set_of_seeds.add(random.randint(num_low, num_high))

    list_of_seeds = list(set_of_seeds)

    # Save seed values
    with open(seeds_file, "w") as f:
        for s in list_of_seeds:
            f.write(str(s) +"\n")
