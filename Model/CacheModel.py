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
import cv2
import numpy
random.seed(1)
torch.manual_seed(1)
from PIL import Image

import gc
from einops import repeat
def ReadImage(path):
    if path.endswith('.npz'):
        image = np.load(path)['image'].astype(np.uint8)
    elif path.endswith('.jpg') | path.endswith('.png'):
        img = Image.open(path)
        image = img.convert('RGB') if len(img.size) == 3 else img.convert('L')
        image = np.array(image)
    else: image = None
    return image
class CacheModel(nn.Module):
    def __init__(self, model, train_loader_cache=None, image_path="",train_conv=False,train_cache=False,using_lora=False, save_or_load_path=None, device='cuda:0'):
        super().__init__()
        self.model = model
        self.mask_size = self.model.sam.image_encoder.img_size // 4
        self.device = device  
        self.cache_loader = train_loader_cache
        # self.dics = self.build_dics()
        self.image_path = image_path
        self.save_or_load_path = save_or_load_path
        self.label_counts = []
        self.image_in_cache = []
        self.ptype_masks, self.ptype_cases,self.ptype_weights = self.build_ptype_masks()
        # np.save(os.path.join(save_or_load_path,'ptype_case.npy') , self.ptype_cases)
        self.train_conv = train_conv
        self.train_cache = train_cache
        self.cache_optimizer= None
        self.scheduler = None
        self.using_lora = using_lora
        
    
        if(train_conv == False):
            with torch.no_grad():
                self.cache_keys = self.build_cache_key(True)
    
    def build_ptype_masks(self):
        if self.cache_loader == None:  # load pretrained ptype_masks
            ptype_masks = torch.load(os.path.join(self.save_or_load_path, 'ptype_masks.pth'), map_location='cpu')
            tmp = torch.sum(ptype_masks, dim=(2, 3)) > 0
            ptype_weight = torch.sum(tmp).item()
            ptype_cases=[]
            print('ptype masks loaded')
        else:
            dics = {}
            for i, (_, img_path, dic, cases,_) in enumerate(self.cache_loader):
                self.image_in_cache.extend(img_path)
                for key, value in dic.items():
                    if int(key) not in dics:
                        dics[int(key)] = []
                    dics[int(key)].extend(value)
            dics = np.array([v for k, v in sorted(dics.items())])
            ptype_masks = torch.zeros((len(dics), len(dics[0]), self.mask_size, self.mask_size)).to(self.device)
            ptype_cases= np.zeros((len(dics), len(dics[0])))
            for i,dic in enumerate(dics):
                count = 0
                
                for j,maskpath in enumerate(dic):
                    mask = ReadImage(os.path.join(self.image_path,maskpath))
                    if mask is not None:
                        ptype_cases[i][j] = int(maskpath.split('/')[-1].split('_')[0])
                        mask = cv2.resize(mask, (self.mask_size, self.mask_size), interpolation=cv2.INTER_CUBIC)
                        ptype_masks[i, j] = torch.from_numpy(mask)
                self.label_counts.append(count)
            tmp = torch.sum(ptype_masks, dim=(2, 3)) > 0
            ptype_weight = torch.sum(tmp).item()
            torch.save(ptype_masks, os.path.join(self.save_or_load_path, 'ptype_masks.pth'))
            
        return ptype_masks , np.array(ptype_cases), ptype_weight
    def build_cache_key(self,isfirst):
        if (self.cache_loader == None):
            cache_keys = torch.load(os.path.join(self.save_or_load_path, 'cache_key.pth'))
            cache_keys = cache_keys.to(self.device)
            print('cache keys loaded')
        else:
            cache_keys = []
            train_features= []
            
            for images, _, _, _ ,_ in self.cache_loader:
                img_embds = self.model.sam.image_encoder(images.to(self.device))
                image_features = self.model.transpose(img_embds)     # image embeddings after transpose
                train_features.append(image_features.view(image_features.size(0), -1))
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys = cache_keys / cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0) #Transpose 
            if self.train_conv == False and self.train_cache == False:
                torch.save(cache_keys, os.path.join(self.save_or_load_path, 'cache_key.pth'))
            if self.train_cache & isfirst:
                adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(self.device)
                adapter.weight = nn.Parameter(cache_keys.t())
                adapter.train()
                return adapter
            # print('cache_keys shape: ', cache_keys.shape)
            
        return cache_keys
    
    def get_mask_logits(self,beta,img_embeds,is_train):
        if((self.train_conv | self.using_lora) & is_train):
            with torch.no_grad():
                self.cache_keys = self.build_cache_key(False)
        if(self.using_lora):
            image_features = self.model.transpose(img_embeds)
        else:
            with torch.no_grad():
                image_features = self.model.transpose(img_embeds)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        affinity =self.cache_keys(image_features) if self.train_cache else torch.mm(image_features, self.cache_keys)
        logits = ((-1) * (beta - beta * affinity)).exp()
        return logits
    
    def forward(self,img_embeds,label,case,beta=10,is_train= False):
        logits = self.get_mask_logits(beta,img_embeds,is_train)
        target_rows = self.ptype_masks[label].to(self.device) 
        if is_train:
            target_cases = self.ptype_cases[label]
        if type(label) == int:
            logits = logits.squeeze(0)
            masks = torch.zeros_like(target_rows).to(self.device)
            logit = torch.where(logits < logits.mean(), logits.min(), logits)
            for j, target in enumerate(target_rows):
                masks[j] = logit[j] * target.to(self.device) 
            masks = torch.sum(masks, axis=0)
            final_masks = (2/3)*masks*1000-5
        else:
            mean_values = logits.mean(dim=1, keepdim=True)
            logits = torch.where(logits < mean_values, logits.min(dim=1, keepdim=True)[0], logits)
            if is_train:
                equal_to_case = (target_cases == case)
                target_rows[equal_to_case] = 0
                logits[equal_to_case] = 0
            logits_expanded = logits.unsqueeze(-1).unsqueeze(-1).to(self.device) 
            masks = (logits_expanded * target_rows)
            final_masks = torch.sum(masks, axis=1) / self.ptype_weights
            final_masks = (2/3)*final_masks*100-5
        return final_masks
    
    def get_image_in_cache(self):
        return self.image_in_cache
    
    def save_cache_keys(self,cache_keys):
         torch.save(cache_keys, os.path.join(self.save_or_load_path, 'cache_key.pth'))