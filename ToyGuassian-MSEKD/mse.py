import os
import torch
import torch.nn.functional as F
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from setting import *
class Centroid():
    def __init__(self, n_classes, Ctemp = 1):
        self.n_classes = n_classes
        self.centroids = torch.ones((n_classes, n_classes)) / n_classes
        self.Ctemp = Ctemp
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    def update_epoch(self, model, data_loader):
        self.centroids = torch.zeros_like(self.centroids)
        model.train()
        device = next(model.parameters()).device
        for image,target, _ in data_loader:
            image,target = image.float().cuda(), target.cuda()
            logit = model(image).detach()

            Classes =  target.cpu().unique()
            logit = logit.cpu()
            output = F.softmax(logit.float(), self.Ctemp)
            
            for Class in Classes:
                # breakpoint()
                mask = (target.cpu() == Class)[:,0]
                # breakpoint()
                self.centroids[int(Class)] += torch.sum(output[mask], axis = 0)

        self.centroids =  self.centroids/(self.centroids.sum(1)[:,None])
        
    def get_centroids(self, target):
        return torch.index_select(self.centroids, 0, target[:,0].cpu().type(torch.int32)).to(target.device)

