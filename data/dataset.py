import numpy as np 
import os
import random 
from PIL import Image  
import torch 
from torch.utils.data import Dataset

from data.transforms import ApplyTransform

class ImageDataset(Dataset):
    def __init__(self, parameters, paths, annotations, species_info, batch_size=1, transform=False, shuffle=False, valid=False):
        self.parameters = {"habitat":parameters[0], "img_size": parameters[1], "path_source_img": parameters[2], "device": parameters[3], "model": parameters[4]}
        self.paths = np.array(paths)
        self.species_info = np.array(species_info)
        self.batch_size = batch_size
        self.transform = transform
        self.order = np.array([i for i in range(len(self.paths))])
        self.shuffle = shuffle
        self.Transfrom_object = ApplyTransform(self.parameters["model"], self.parameters["img_size"]) 
        self.valid = valid


        annotations = np.array(annotations)
        self.labels = np.array([1 if self.parameters["habitat"] in np.where(label_vec == 1)[0] else 0 for label_vec in annotations])
        self.class0_indices = np.where(self.labels == 0)[0].tolist()
        self.class1_indices = np.where(self.labels == 1)[0].tolist()


        if not self.valid : 
            self.num_batches = min(len(self.class0_indices), len(self.class1_indices)) * 2 // self.batch_size
        else :
            self.num_batches = (len(self.labels) + self.batch_size - 1) // self.batch_size



    def __len__(self):
        return self.num_batches
    
    def load_img(self,im_path):
        im = Image.open(os.path.join(self.parameters["path_source_img"],im_path)).convert('RGB')
        return Image.fromarray(np.array(im)[:, :, ::-1])

    def __getitem__(self, idx):
        if not self.valid : 
            half_batch = self.batch_size // 2
            if self.shuffle:
                random.shuffle(self.class0_indices)
                random.shuffle(self.class1_indices)

            class0_sample = self.class0_indices[idx * half_batch:(idx + 1) * half_batch]
            class1_sample = self.class1_indices[idx * half_batch:(idx + 1) * half_batch]

            sample_indices = class0_sample + class1_sample
            if self.shuffle:
                random.shuffle(sample_indices)

            
            labels = self.labels[sample_indices]
            paths = self.paths[sample_indices]
            species_info = self.species_info[sample_indices]
            images = [self.load_img(p) for p in paths]
            images = [self.Transfrom_object.resize(im) for im in images]
            images = [self.Transfrom_object.augment(im) for im in images]
            images = [self.Transfrom_object.normalise(im).unsqueeze(0) for im in images]

            return torch.stack(images), torch.tensor(labels, dtype=torch.long), paths, species_info
        
        else : 

            order = self.order[idx*self.batch_size:(idx*self.batch_size)+self.batch_size]

            labels = self.labels[order]
            paths = np.array(order)
            species_info = self.species_info[order]
            images = [self.load_img(p) for p in self.paths[order]]
            images = [self.Transfrom_object.resize(im) for im in images]
            images = [self.Transfrom_object.augment(im) for im in images]
            images = torch.stack([self.Transfrom_object.normalise(im).unsqueeze(0) for im in images])

            return images, torch.tensor(labels, dtype=torch.long), paths, species_info