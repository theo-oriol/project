import os 
import random
import pickle
import pandas as pd
import numpy as np 

from utils.utils import load_config
from data.utils import species_name_extraction

parameters = load_config()

path_source_img = parameters["path_source_img"]

unique_species_name_list, all_species_info = species_name_extraction(parameters)


random.shuffle(unique_species_name_list)
train_species = unique_species_name_list[:int(len(unique_species_name_list)*0.8)]
valid_species = unique_species_name_list[int(len(unique_species_name_list)*0.8):]

with open(f"saved_split_limit:{parameters['image_limitation']}.pkl", "wb") as f:
    pickle.dump({"train_species": train_species, "valid_species": valid_species}, f)