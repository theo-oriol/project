import argparse
import torch
from torch.utils.data import DataLoader


from data.dataset import ImageDataset
from data.health import check
from data.utils import extract_labels, extract_labels_and_image, species_name_extraction, startup_dir
from model.dino_model import Classifier
from utils.tsne import tsne
from utils.plot import auc_plot, families_plot, pression_recall, species_plot
from utils.utils import load_species_split, predictions_last_epochs, report

def eval(parameters):
    
    destination_dir = startup_dir(parameters)

    model = Classifier()
    model.load_state_dict(torch.load("/home/oriol@newcefe.newage.fr/Models/project/results/Dino_binary_sampling_1000_13_6/model", weights_only=True))
    model.eval()
    device = torch.device(parameters["device"])
    model.to(device)

    unique_species_name_list, all_species_info = species_name_extraction(parameters)
    train_species,valid_species = load_species_split(parameters)

    labels_in_csv, families, env_names, habitats = extract_labels(parameters,unique_species_name_list)
    (list_of_train_image_labels, list_of_train_image_path, list_of_train_image_info), (list_of_valid_image_labels, list_of_valid_image_path, list_of_valid_image_info), list_of_family_id, list_of_species_id = extract_labels_and_image(all_species_info, labels_in_csv, families, (train_species,valid_species))

    check((list_of_train_image_labels, list_of_train_image_info),(list_of_valid_image_labels, list_of_valid_image_info),parameters["habitat"], habitats[parameters["habitat"]],destination_dir)

    valid_dataset = ImageDataset(parameters,list_of_valid_image_path, list_of_valid_image_labels, list_of_valid_image_info, batch_size=parameters["valid_batch_size"], valid=True)
    valid_loader = DataLoader(valid_dataset, batch_size=None, shuffle=False)

    train_dataset = ImageDataset(parameters,list_of_train_image_path, list_of_train_image_labels, list_of_train_image_info, batch_size=parameters["valid_batch_size"], valid=True)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False)

    (all_train_preds,all_train_labels,all_train_features,all_train_real_prob,all_train_species,all_train_sexe,all_train_family), (all_valid_preds,all_valid_labels,all_valid_features,all_valid_real_prob,all_valid_species,all_valid_sexe,all_valid_family) = predictions_last_epochs(parameters,model,(train_loader,valid_loader))

    report(all_valid_labels, all_valid_preds,destination_dir)
    pression_recall(all_valid_real_prob,all_valid_labels,habitats[parameters["habitat"]],destination_dir)
    families_plot(all_valid_real_prob,all_valid_labels,all_valid_family,habitats[parameters["habitat"]],families,destination_dir)
    species_plot(all_valid_real_prob,all_valid_labels,all_valid_species,destination_dir)
    auc_plot(all_valid_real_prob,all_valid_labels,destination_dir)
    tsne(all_valid_features,all_valid_labels,destination_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", "-n", type=str, required=True, help="Name of dir")
    parser.add_argument("--model","-m", type=str, choices=["dino", "None"], default="dino", help="model type")
    parser.add_argument("--valid_batch_size", "-b", type=int, default=40, help="batch size")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--habitat", "-hb", type=int, default=13, help="habitat selected")
    parser.add_argument("--path_model", "-pm", type=str, required=True, help="model weights path")
    parser.add_argument("--path_source_csv","-psc", type=str, default="/home/oriol@newcefe.newage.fr/Datasets/final_table2.csv", help="path to csv file")
    parser.add_argument("--path_source_img","-psi", type=str, default="/home/oriol@newcefe.newage.fr/Datasets//whole_bird", help="path to image folder")
    parser.add_argument("--path_source_split","-pss", type=str, default="/home/oriol@newcefe.newage.fr/Models/project/saved_split_limit:None.pkl", help="path to split file")
    parser.add_argument("--img_size", "-size", type=int, default=224, help="image size")
    parser.add_argument("--img_limitation","-iml", type=int, default=None, help="image limitation (None for no limitation)")

    args = parser.parse_args()

    eval(vars(args))