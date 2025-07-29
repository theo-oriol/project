import os 
import pandas as pd 
import numpy as np 

def startup_dir(parameters):
    """
    Create the directory that wil contain the results of the training
    return path of the directory
    """
    print("Creating destination Directory")
    i=0
    while os.path.isdir(os.path.join("results",f"{parameters['name']}_{parameters['habitat']}_{i}")) : i+=1
    destination_dir = os.path.join("results",f"{parameters['name']}_{parameters['habitat']}_{i}")
    os.makedirs(destination_dir)
    return destination_dir


def species_name_extraction(parameters) :
    """
    Extract the name of the species from the path of the images and return the list of unique name and info about each image (path, sexe and species)
    """

    if not parameters["image_limitation"]:
        list_of_images_path  = [ file  for file in os.listdir(parameters["path_source_img"]) if "png" in file]
    else : 
        list_of_images_path  = [ file  for file in os.listdir(parameters["path_source_img"]) if "png" in file][:parameters["image_limitation"]]
    all_species_info = {} # toutes les espèces
    for image_url in list_of_images_path:
        path_split = image_url.split("_")
        path_species = " ".join(path_split[:2])
        path_sexe = 0 if path_split[3] == "F" else 1
        if path_species not in all_species_info : all_species_info[path_species]=[]
        all_species_info[path_species].append([image_url,path_sexe])

    unique_species_name_list = list({d for d in all_species_info}) 
    return unique_species_name_list, all_species_info





def extract_labels(parameters,unique_species_name_list):
    """
    Extract info about species from the csv file
    return the list of species and the list of labels
    """
    df = pd.read_csv(parameters["path_source_csv"])
    df["sp_image"] = df["sp_image"]
    # species_list = [s.lower().strip() for s in unique_species_name_list]
    species_list = [s for s in unique_species_name_list]
    df_filtered = df[df["sp_image"].isin(species_list)]
    habitat_columns = df.columns[5:]
    habitats = list(habitat_columns)
    df_tags = df_filtered[["sp_image"] + list(habitat_columns)]
    labels = df_tags.set_index("sp_image").astype(int).apply(list, axis=1).to_dict()
    env_names_clean = [col for col in habitat_columns]
    family_columns = df.columns[1]
    df_fam = df_filtered[["sp_image"] + [family_columns]]
    families = df_fam.set_index("sp_image").to_dict()["family"]

    return labels, families, env_names_clean, habitats


def extract_labels_and_image(all_species_name, labels_in_csv, family, split):
    """
    Extract image info and split to build a list of image path and labels (species, env, sexe)
    return the list of image path and labels for train and valid
    """
    train_species,valid_species = split

    list_of_train_image_labels = []
    list_of_train_image_info = []
    list_of_train_image_path = []
    
    list_of_valid_image_labels = []
    list_of_valid_image_info = []
    list_of_valid_image_path = []

    list_of_species_id = {}
    list_of_family_id = {}
    i=0
    i2=0
    for species in all_species_name.keys():
            if species not in labels_in_csv: continue # Certains espèces ne se retrouve pas dans le csv
            if species not in list_of_species_id:
                 list_of_species_id[species] = i
                 i+=1
            
            fam = family[species]

            if fam not in list_of_family_id:
                 list_of_family_id[fam] = i2
                 i2+=1

            if species in train_species:
                for indiv in all_species_name[species]:
                    list_of_train_image_labels.append(labels_in_csv[species])
                    list_of_train_image_path.append(indiv[0])
                    list_of_train_image_info.append((list_of_species_id[species],indiv[1],list_of_family_id[fam]))
            elif species in valid_species:             
                for indiv in all_species_name[species]:
                    list_of_valid_image_labels.append(labels_in_csv[species])
                    list_of_valid_image_path.append(indiv[0])
                    list_of_valid_image_info.append((list_of_species_id[species],indiv[1],list_of_family_id[fam]))


    return (list_of_train_image_labels, list_of_train_image_path, list_of_train_image_info), (list_of_valid_image_labels, list_of_valid_image_path, list_of_valid_image_info), list_of_family_id, list_of_species_id


