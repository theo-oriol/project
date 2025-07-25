from collections import defaultdict
import torch 
import numpy as np 

class metrics_saver():
    def __init__(self):
        self.metrics = {
                "species_macro_accuracy":[],
                "sexe_macro_accuracy":[],
                "macro_accuracy":[],
                "accuracy":[],
                "loss":[],
            }
        

    def init_buffer(self):
        self.buffer_species_macro_accuracy_preds = defaultdict(list)
        self.buffer_species_macro_accuracy_labels = defaultdict(list)
        #####################################
        self.buffer_sexe_macro_accuracy_preds = defaultdict(list)
        self.buffer_sexe_macro_accuracy_labels = defaultdict(list)
        #####################################
        self.buffer_macro_accuracy_preds = []
        self.buffer_macro_accuracy_labels = []
        #####################################
        self.buffer_accuracy = 0
        self.bugger_accuracy_total = 0
        #####################################
        self.buffer_loss = []

    def merge(self):
        self.species_macro_accuracy()
        self.sexe_macro_accuracy()
        self.macro_accuracy()
        self.accuracy()
        self.loss()

    def update(self, y_pred,y_true,loss,info):        
        
        self.buffering_loss(loss)

        preds = (torch.sigmoid(y_pred) > 0.5).int()
        self.buffering_species_macro_accuracy(preds,y_true,info)
        self.buffering_sexe_macro_accuracy(preds,y_true,info)
        self.buffering_macro_accuracy(preds,y_true)
        self.buffering_accuracy(preds,y_true)

    #################################################
    def buffering_species_macro_accuracy(self,y_pred,y_true,info):
        for sid, pred, label in zip(info, y_pred, y_true.int()):
            self.buffer_species_macro_accuracy_preds[sid[0].item()].append(pred.item())
            self.buffer_species_macro_accuracy_labels[sid[0].item()].append(label.item())
        

    def species_macro_accuracy(self):
        species_acc = {}
        for sid in self.buffer_species_macro_accuracy_preds:
            preds = torch.tensor(self.buffer_species_macro_accuracy_preds[sid])
            labels = torch.tensor(self.buffer_species_macro_accuracy_labels[sid])
            acc = (preds == labels).sum().item() / len(labels)
            species_acc[sid] = acc
        self.metrics["species_macro_accuracy"].append(species_acc)
    #################################################
    def buffering_sexe_macro_accuracy(self,y_pred,y_true,info):
        for sid, pred, label in zip(info, y_pred, y_true.int()):
            self.buffer_sexe_macro_accuracy_preds[sid[1].item()].append(pred.item())
            self.buffer_sexe_macro_accuracy_labels[sid[1].item()].append(label.item())

    def sexe_macro_accuracy(self):
        sexe_acc = {}
        for sid in self.buffer_sexe_macro_accuracy_preds:
            preds = torch.tensor(self.buffer_sexe_macro_accuracy_preds[sid])
            labels = torch.tensor(self.buffer_sexe_macro_accuracy_labels[sid])
            acc = (preds == labels).sum().item() / len(labels)
            sexe_acc[sid] = acc
        self.metrics["sexe_macro_accuracy"].append(sexe_acc)
    #################################################
    def buffering_macro_accuracy(self,y_pred,y_true):
        self.buffer_macro_accuracy_preds.extend(y_pred.cpu().numpy())
        self.buffer_macro_accuracy_labels.extend(y_true.int().cpu().numpy())

    
    def macro_accuracy(self):
        labels_tensor = torch.tensor(self.buffer_macro_accuracy_labels)
        preds_tensor = torch.tensor(self.buffer_macro_accuracy_preds)
        macro_acc = 0
        for cls in [0, 1]:
            cls_mask = labels_tensor == cls
            if cls_mask.sum() > 0:
                cls_acc = (preds_tensor[cls_mask] == labels_tensor[cls_mask]).float().mean().item()
                macro_acc += cls_acc
        macro_acc /= 2
        self.metrics["macro_accuracy"].append(macro_acc)
    #################################################
    def buffering_accuracy(self,y_pred,y_true):
        self.buffer_accuracy += (y_pred == y_true.int()).sum().item()
        self.bugger_accuracy_total += y_true.size(0)

    def accuracy(self):
        accuracy = self.buffer_accuracy / self.bugger_accuracy_total
        self.metrics["accuracy"].append(accuracy)
    #################################################
    def buffering_loss(self,l):
        self.buffer_loss.append(l)

    def loss(self):
        l = np.mean(self.buffer_loss)
        self.metrics["loss"].append(l)
        