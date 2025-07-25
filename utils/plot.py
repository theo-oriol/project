import numpy as np 
import os 
import matplotlib.pyplot as plt
import colorsys
import random
import torch 
from torcheval.metrics import BinaryPrecisionRecallCurve

def plot_loss(parameters,metrics,destination_dir):
    (train_metrics, valid_metrics) = metrics


    train_macro_species_acc = []
    for i in range(len(train_metrics.metrics["species_macro_accuracy"])):
        train_macro_species_acc.append(np.mean(list(train_metrics.metrics["species_macro_accuracy"][i].values())))

    valid_macro_species_acc = []
    for i in range(len(valid_metrics.metrics["species_macro_accuracy"])):
        valid_macro_species_acc.append(np.mean(list(valid_metrics.metrics["species_macro_accuracy"][i].values())))

    train_acc_female = []
    train_acc_male = []
    for i in range(len(train_metrics.metrics["sexe_macro_accuracy"])):
        train_acc_female.append(train_metrics.metrics["sexe_macro_accuracy"][i][0])
        train_acc_male.append(train_metrics.metrics["sexe_macro_accuracy"][i][1])


    valid_acc_female = []
    valid_acc_male = []
    for i in range(len(valid_metrics.metrics["sexe_macro_accuracy"])):
        valid_acc_female.append(valid_metrics.metrics["sexe_macro_accuracy"][i][0])
        valid_acc_male.append(valid_metrics.metrics["sexe_macro_accuracy"][i][1])

    epochs = [i for i in range(parameters["epochs"])]
    
    plt.figure(figsize=(18, 10))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_metrics.metrics["loss"], label="Train Loss")
    plt.plot(epochs, valid_metrics.metrics["loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_metrics.metrics["accuracy"], label="Train Accuracy")
    plt.plot(epochs, valid_metrics.metrics["accuracy"], label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(epochs, train_metrics.metrics["macro_accuracy"], label="Train Macro Accuracy")
    plt.plot(epochs, valid_metrics.metrics["macro_accuracy"], label="Valid Macro Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Macro Accuracy")
    plt.title("Macro Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(epochs, train_macro_species_acc, label="Train Macro SPE Accuracy")
    plt.plot(epochs, valid_macro_species_acc, label="Valid Macro SPE Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Macro Species Accuracy")
    plt.title("Macro Species Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(epochs, train_acc_female, label="Train female Accuracy")
    plt.plot(epochs, train_acc_male, label="Train male Accuracy")
    plt.plot(epochs, valid_acc_female, label="Valid female Accuracy")
    plt.plot(epochs, valid_acc_male, label="Valid male Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Sexe Accuracy")
    plt.title("Sexe Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir,"losses"))






def pression_recall(all_valid_real_prob,all_valid_labels,habitat,destination_dir):
    metric = BinaryPrecisionRecallCurve()
    metric.update(torch.from_numpy(all_valid_real_prob), torch.from_numpy(all_valid_labels))
    precision_list, recall_list, _ = metric.compute()

    precision = np.array(precision_list)
    recall = np.array(recall_list)

    def generate_n_colors(n):
        hues = np.linspace(0, 1, n, endpoint=False)
        saturations = np.linspace(0.6, 0.9, n)
        values = np.linspace(0.8, 1.0, n)

        colors = [
            colorsys.hsv_to_rgb(h, s, v)
            for h, s, v in zip(hues, saturations, values)
        ]
        return colors

    colors = generate_n_colors(len(precision))
    random.shuffle(colors)

    plt.figure(figsize=(10,5))

    plt.plot(recall,precision, label=f"{habitat} {np.mean(precision):.3f}")

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.ylim(0, 1)
    plt.legend(title="Class",loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(destination_dir,"PR"))


def families_plot(all_valid_real_prob,all_valid_labels,all_valid_family,habitat,families,destination_dir):

    ap_per_env_per_family = []
    img_per_family = []
    img_per_env_per_family = []

    for i in np.unique(all_valid_family):
        mask = all_valid_family == i

        y_true = torch.from_numpy(all_valid_labels[mask])
        y_pred = torch.from_numpy(all_valid_real_prob[mask])

        metric = BinaryPrecisionRecallCurve()
        metric.update(y_pred, y_true)
        precision, recall, _ = metric.compute()

        precision = np.array(precision)
        recall = np.array(recall)

        ap_per_env = []


        rec = recall
        prec = precision
        sorted_idx = np.argsort(rec)
        rec = rec[sorted_idx]
        prec = prec[sorted_idx]
        ap = np.trapz(prec, rec)
        ap_per_env.append(ap)

        ap_per_env_per_family.append(ap_per_env)
        img_per_family.append(len(y_true))
        img_per_env_per_family.append(np.sum(all_valid_labels[mask], axis=0))


    ap_per_env_per_family = np.array(ap_per_env_per_family)  # shape: [n_families, n_envs]
    mAP_per_family = np.mean(ap_per_env_per_family, axis=1)
    img_per_env_per_family = np.array(img_per_env_per_family)

    plt.figure()
    plt.hist(mAP_per_family, bins=22, range=(0,1), alpha=0.8, color='steelblue')
    plt.title(f"distribution of mAP per family")
    plt.xlabel("mAP")
    plt.ylabel("Count")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir,"mAP per family"))

    plt.figure()
    plt.plot(img_per_env_per_family, ap_per_env_per_family, 'bo', markersize=3)
    plt.xlabel("Count")
    plt.ylabel("AP (per family)")
    plt.grid(True)

    plt.title("Distribution of AP per Family", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.savefig(os.path.join(destination_dir,"Distribution of AP per Family"))