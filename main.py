import os 
import torch
from torch.utils.data import DataLoader

from utils.tsne import tsne
from utils.utils import load_config, load_species_split, predictions_last_epochs,save_config,report,save_log
from utils.plot import families_plot, plot_loss, pression_recall
from data.utils import species_name_extraction, extract_labels, extract_labels_and_image, startup_dir
from data.dataset import ImageDataset
from data.health import check
from model.dino_model import Classifier
from train.losses import get_loss_function
from train.scheduler import get_optimizer, get_scheduler
from train.trainer import train
from train.metrics import metrics_saver


parameters = load_config()

destination_dir = startup_dir(parameters)

save_config(parameters,destination_dir)

unique_species_name_list, all_species_info = species_name_extraction(parameters)
train_species,valid_species = load_species_split(parameters)

labels_in_csv, families, env_names, habitats = extract_labels(parameters,unique_species_name_list)
(list_of_train_image_labels, list_of_train_image_path, list_of_train_image_info), (list_of_valid_image_labels, list_of_valid_image_path, list_of_valid_image_info), list_of_family_id, list_of_species_id = extract_labels_and_image(all_species_info, labels_in_csv, families, (train_species,valid_species))

check((list_of_train_image_labels, list_of_train_image_info),(list_of_valid_image_labels, list_of_valid_image_info),parameters["env_selected"], habitats[parameters["env_selected"]],destination_dir)


train_dataset = ImageDataset(parameters,list_of_train_image_path, list_of_train_image_labels, list_of_train_image_info, batch_size=parameters["train_batch_size"])
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

valid_dataset = ImageDataset(parameters,list_of_valid_image_path, list_of_valid_image_labels, list_of_valid_image_info, batch_size=parameters["valid_batch_size"], valid=True)
valid_loader = DataLoader(valid_dataset, batch_size=None, shuffle=False)

model = Classifier()

device = torch.device(parameters["device"])

model.to(device)

criterion = get_loss_function(parameters["loss"])
optimizer = get_optimizer(parameters,model)
scheduler = get_scheduler(parameters,optimizer)

trained_model, metrics, complete_log = train(parameters,(train_loader,valid_loader),model,criterion,optimizer,scheduler,metrics_saver)

save_log(complete_log,destination_dir)
torch.save(trained_model.state_dict(),os.path.join(destination_dir,"model"))  


train_dataset = ImageDataset(parameters,list_of_train_image_path, list_of_train_image_labels, list_of_train_image_info, batch_size=parameters["train_batch_size"], valid=True)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False)

(all_train_preds,all_train_labels,all_train_features,all_train_real_prob,all_train_sexe,all_train_family), (all_valid_preds,all_valid_labels,all_valid_features,all_valid_real_prob,all_valid_sexe,all_valid_family) = predictions_last_epochs(parameters,model,(train_loader,valid_loader))


report(all_valid_labels, all_valid_preds,destination_dir)

plot_loss(parameters,metrics,destination_dir)

pression_recall(all_valid_real_prob,all_valid_labels,habitats[parameters["env_selected"]],destination_dir)
families_plot(all_valid_real_prob,all_valid_labels,all_valid_family,habitats[parameters["env_selected"]],families,destination_dir)

tsne(all_valid_features,all_valid_labels,destination_dir)