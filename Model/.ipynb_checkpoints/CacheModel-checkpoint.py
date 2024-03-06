from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pynvml import *
from .segment_anything import sam_model_registry
import random
from scipy.ndimage import zoom
import os

random.seed(1)
torch.manual_seed(1)

import gc
from einops import repeat

class CacheModel():
    def __init__(self, model, train_loader_cache=None, image_path="",train_conv=False,train_cache=False, save_or_load_path=None, device='cuda:0'):
        self.model =  model
        self.mask_size = self.model.sam.image_encoder.img_size // 4
        self.device = device  
        self.cache_loader = train_loader_cache
        # self.dics = self.build_dics()
        self.image_path = image_path
        self.save_or_load_path = save_or_load_path
        self.ptype_masks = self.build_ptype_masks()
        self.train_conv = train_conv
        self.train_cache = train_cache
        self.cache_optimizer= None
        self.scheduler = None
        if(train_conv == False):
            with torch.no_grad():
                self.cache_keys = self.build_cache_key(True)
    
    def build_ptype_masks(self):
        if self.cache_loader == None:  # load pretrained ptype_masks
            ptype_masks = torch.load(os.path.join(self.save_or_load_path, 'ptype_masks.pth'))
            print('ptype masks loaded')
        else:
            dics = {}
            for i, (_,dic,_) in enumerate(self.cache_loader):
                for key, value in dic.items():
                    if int(key) not in dics:
                        dics[int(key)] = []
                    dics[int(key)].extend(value)
            dics = np.array([v for k, v in sorted(dics.items())])
            ptype_masks = torch.zeros((len(dics), len(dics[0]), self.mask_size, self.mask_size)).to(self.device)
            for i,dic in enumerate(dics):
                for j,maskpath in enumerate(dic):
                    if maskpath.endswith('npz'):
                        mask = np.load(os.path.join(self.image_path,maskpath))['image']
                        mask = zoom(mask, (self.mask_size/mask.shape[0], self.mask_size /mask.shape[0]), order=0)
                        ptype_masks[i, j] = torch.from_numpy(mask)
            torch.save(ptype_masks, os.path.join(self.save_or_load_path, 'ptype_masks.pth'))
            
        return ptype_masks

    def build_cache_key(self,isfirst):
        if (self.cache_loader == None):
            cache_keys = torch.load(os.path.join(self.save_or_load_path, 'cache_key.pth'))
            print('cache keys loaded')
        else:
            cache_keys = []
            train_features= []
            for images, _, _ in self.cache_loader:
                with torch.no_grad():
                    img_embds = self.model.sam.image_encoder(images.to(self.device))
                image_features = self.model.transpose(img_embds)     # image embeddings after transpose
                train_features.append(image_features.view(image_features.size(0), -1))
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys = cache_keys / cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0) #Transpose 
            if self.train_conv == False:
                self.save_cache_keys(cache_keys)
            if self.train_cache & isfirst:
                adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(self.device)
                adapter.weight = nn.Parameter(cache_keys.t())
                adapter.train()
                return adapter
        return cache_keys
    
    def get_mask_logits(self,beta,img_embds,is_train):
        if(self.train_conv & is_train):
            self.cache_keys = self.build_cache_key(False)
        if(self.train_conv):
            image_features = self.model.transpose(img_embds)
        else:
            with torch.no_grad():
                image_features = self.model.transpose(img_embds)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        affinity =self.cache_keys(image_features) if self.train_cache else torch.mm(image_features,self.cache_keys)
        logits = ((-1) * (beta - beta * affinity)).exp()
        return logits
    
    def get_prompt_masks(self,img_embeds,label,beta=50,is_train= False):
        logits = self.get_mask_logits(beta,img_embeds,is_train)
        target_rows = self.ptype_masks[label] 
        masks = torch.zeros((len(target_rows), len(target_rows[0]), self.mask_size, self.mask_size)).to(self.device)
        for i, target_row in enumerate(target_rows):
            logit = torch.where(logits[i] < logits[i].mean(), logits[i].min(), logits[i])
            for j, target in enumerate(target_row):
                masks[i, j] = logit[j] * target.to(self.device)
        final_masks = torch.sum(masks, axis=1)
        for i,mask in enumerate(torch.sum(masks, axis=1)):
            if((mask.max()-mask.min()) !=0):
                mask = (mask- mask.min()) / (mask.max()-mask.min())
            final_masks[i] = mask*40-20
        return final_masks
    def save_cache_keys(self,cache_keys):
         torch.save(cache_keys, os.path.join(self.save_or_load_path, 'cache_key.pth'))