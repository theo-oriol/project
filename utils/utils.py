import torch 
import numpy as np 
import os 
import yaml
import pickle
from sklearn.metrics import classification_report

def predictions_last_epochs(device,model,dataloader):
    train_loader,valid_loader = dataloader
    model.eval()

    all_train_preds = []
    all_train_labels = []
    all_train_features = []
    all_train_real_prob = []
    all_train_species = []
    all_train_sexe = []
    all_train_family = []
    
    with torch.no_grad():
            for i, (inputs, labels, _,info) in enumerate(train_loader):
                inputs = inputs.squeeze().to(device)

                
                if inputs.ndim == 3:
                    inputs = inputs.unsqueeze(0)

                outputs, features = model(inputs)
                outputs = outputs.squeeze()

                preds = (torch.sigmoid(outputs) > 0.5).int()


                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels)
                all_train_features.extend(features.cpu().numpy())
                all_train_real_prob.extend(torch.sigmoid(outputs).cpu().numpy())
                all_train_species.extend(info[:,0])
                all_train_sexe.extend(info[:,1])
                all_train_family.extend(info[:,2].cpu().numpy())

    all_train_preds = np.array(all_train_preds)
    all_train_labels = np.array(all_train_labels)
    all_train_features = np.array(all_train_features)
    all_train_real_prob = np.array(all_train_real_prob)
    all_train_species = np.array(all_train_species)
    all_train_sexe = np.array(all_train_sexe)
    all_train_family = np.array(all_train_family)

    all_valid_preds = []
    all_valid_labels = []
    all_valid_features = []
    all_valid_real_prob = []
    all_valid_species = []
    all_valid_sexe = []
    all_valid_family = []

    with torch.no_grad():
            for inputs, labels, _, info in valid_loader:

                inputs = inputs.squeeze().to(device)

                if inputs.ndim == 3:
                    inputs = inputs.unsqueeze(0)

                outputs, features = model(inputs)
                outputs = outputs.squeeze()

                preds = (torch.sigmoid(outputs) > 0.5).int()

                all_valid_preds.extend(preds.cpu().numpy())
                all_valid_labels.extend(labels)
                all_valid_features.extend(features.cpu().numpy())
                all_valid_real_prob.extend(torch.sigmoid(outputs).cpu().numpy())
                all_valid_species.extend(info[:,0])
                all_valid_sexe.extend(info[:,1])
                all_valid_family.extend(info[:,2].cpu().numpy())
                
    all_valid_preds = np.array(all_valid_preds)
    all_valid_labels = np.array(all_valid_labels)
    all_valid_features = np.array(all_valid_features)
    all_valid_real_prob = np.array(all_valid_real_prob)
    all_valid_species = np.array(all_valid_species)
    all_valid_sexe = np.array(all_valid_sexe)
    all_valid_family = np.array(all_valid_family)

    return (all_train_preds,all_train_labels,all_train_features,all_train_real_prob,all_train_species,all_train_sexe,all_train_family), (all_valid_preds,all_valid_labels,all_valid_features,all_valid_real_prob,all_valid_species,all_valid_sexe,all_valid_family)


def report(all_valid_labels, all_valid_preds,destination_dir):
    report_val = classification_report(all_valid_labels, all_valid_preds, zero_division=0)
    with open(os.path.join(destination_dir,"classification_report_valid" ), "w") as f:
        f.write(report_val)

def save_log(log,destination_dir):
    with open(os.path.join(destination_dir,"log.txt"), "w") as f:
        f.write(log)

def save_config(parameters,destination_dir):
    with open(os.path.join(destination_dir,"config.yaml"), 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)

def load_config():
    from config import parameters
    return parameters

def load_species_split(path_source_split):
    """
    Load the file containing the split (species in train/ species in val)
    return the split
    """
    with open(path_source_split, "rb") as f:
        split = pickle.load(f)

    train_species = split["train_species"]
    valid_species = split["valid_species"]

    return (train_species, valid_species)