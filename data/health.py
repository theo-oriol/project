import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def check(train,valid,habitat_selected, habitat_name,destination_dir):
    (list_of_train_image_labels, list_of_train_image_info) = train 
    (list_of_valid_image_labels, list_of_valid_image_info) = valid

    list_of_train_image_sexe, list_of_valid_image_sexe = [],[]
    for i in range(len(list_of_train_image_info)):
        list_of_train_image_sexe.append(list_of_train_image_info[i][1])
    for i in range(len(list_of_valid_image_info)):
        list_of_valid_image_sexe.append(list_of_valid_image_info[i][1])

    list_of_image_species = []
    for i in range(len(list_of_train_image_info)):
        list_of_image_species.append(list_of_train_image_info[i][0])
    for i in range(len(list_of_valid_image_info)):
        list_of_image_species.append(list_of_valid_image_info[i][0])

    list_of_image_family = []
    for i in range(len(list_of_train_image_info)):
        list_of_image_family.append(list_of_train_image_info[i][2])
    for i in range(len(list_of_valid_image_info)):
        list_of_image_family.append(list_of_valid_image_info[i][2])

    plot_species_balance(list_of_image_species, destination_dir)
    plot_family_balance(list_of_image_family, destination_dir)
    plot_class_distribution(habitat_name, list_of_train_image_labels, destination_dir)
    plot_distribution((list_of_train_image_labels, list_of_valid_image_labels), (list_of_train_image_sexe, list_of_valid_image_sexe), habitat_selected, destination_dir)



def plot_species_balance(species, destination_dir):
    species_counts = np.zeros(np.max(species)+1)
    for s in species:
        species_counts[s] += 1


    plt.figure()
    plt.hist(species_counts, bins=4, edgecolor='black')
    plt.ylabel('Number of Samples')
    plt.title('Species Balance')
    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir, "Species_balance"))
    plt.close()

def plot_family_balance(families, destination_dir):
    family_counts = np.zeros(np.max(families)+1)
    for f in families:
        family_counts[f] += 1

    plt.figure(figsize=(12, 8))
    plt.hist(family_counts, bins=20, edgecolor='black')
    plt.xlabel('Number of Samples per Family')
    plt.ylabel('Frequency')
    plt.title('Family Sample Count Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir, "Family_distribution"))


def plot_class_distribution(habitat_selected,labels,destination_dir):
    class_counts = np.zeros(2)
    for l in labels:
        class_counts[l] += 1

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(class_counts)), class_counts)
    plt.xticks(range(len(class_counts)), ['Class 0', 'Class 1'])
    plt.ylabel('Number of Samples')
    plt.title(f'Class Distribution for {habitat_selected}')
    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir, "Class_distribution"))


def plot_distribution(labels, sexe, habitat_selected, destination_dir):
    all_train_labels, all_valid_labels = labels
    num_classes = 2  

    train_class_counts = np.zeros(num_classes)
    valid_class_counts = np.zeros(num_classes)

    for l  in all_train_labels:
        if l[habitat_selected] == 1:
            train_class_counts[1] += 1
        else:
            train_class_counts[0] += 1

    for l in all_valid_labels:
        if l[habitat_selected] == 1:
            valid_class_counts[1] += 1
        else:
            valid_class_counts[0] += 1

    train_counts = train_class_counts[1]
    valid_counts = valid_class_counts[1]
    class_names = [f"Class {i}" for i in range(1)]

    x = np.arange(1)
    width = 0.15



    plt.figure(figsize=(10, 8))
    plt.bar(x - width/2, train_counts/len(all_train_labels), width, label='Train')
    plt.bar(x + width/2, valid_counts/len(all_valid_labels), width, label='Validation')


    plt.text(x-width/2, (train_counts/len(all_train_labels)) , str(train_counts)+"/"+str(len(all_train_labels))+ f" ({train_counts/len(all_train_labels)})", ha='center', va='bottom',fontsize=8)
    plt.text(x+width/2,(valid_counts/len(all_valid_labels)), str(valid_counts)+"/"+str(len(all_valid_labels))+ f" ({valid_counts/len(all_valid_labels)})", ha='center', va='bottom',fontsize=8)

    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Train and Validation Sets')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir,"Class_distribution"))


    all_train_sexe, all_valid_sexe = sexe

    train_sexe_counts = np.zeros(num_classes)
    valid_sexe_counts = np.zeros(num_classes)

    for l  in all_train_sexe:
        train_sexe_counts[l] += 1
    for l in all_valid_sexe:
        valid_sexe_counts[l] += 1
    
    train_sexe_counts = train_sexe_counts[1]
    valid_sexe_counts = valid_sexe_counts[1]


    plt.figure(figsize=(10, 8))
    plt.bar(x - width/2, train_sexe_counts/len(all_train_sexe), width, label='Train')
    plt.bar(x + width/2, valid_sexe_counts/len(all_valid_sexe), width, label='Validation')


    plt.text(x-width/2, (train_sexe_counts/len(all_train_sexe)), str(train_sexe_counts)+"/"+str(len(all_train_sexe))+ f" ({train_sexe_counts/len(all_train_sexe)})", ha='center', va='bottom',fontsize=8)
    plt.text(x+width/2, (valid_sexe_counts/len(all_valid_sexe)), str(valid_sexe_counts)+"/"+str(len(all_valid_sexe))+ f" ({valid_sexe_counts/len(all_valid_sexe)})", ha='center', va='bottom',fontsize=8)

    plt.ylabel('Number of Samples')
    plt.title('Male Distribution in Train and Validation Sets')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir,"Sexe_distribution"))