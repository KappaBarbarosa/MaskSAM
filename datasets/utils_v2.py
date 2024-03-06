import os
import random
import os.path as osp
from collections import defaultdict
import json
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import zoom
from einops import repeat
import cv2
import copy
import time
def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj
def read_image(path,istest=False,isGT=False):
    if(type(path) != str):
        return path
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))
    if path.endswith('.npz'):
        d = np.load(path)
        img = d['image']
    else:
        img = Image.open(path).convert('L') if isGT else Image.open(path).convert('RGB') 
#         if len(img.size)==3:
#             img = img.convert('RGB') 
#         else:
#             img = img.convert('L') 
        img = np.array(img)
    return img


class Datum:
    def __init__(self, impath='', label=0, maskpath="",CN=0,SN=None):
        self.impath = impath
        self.label = label
        self.maskpath = maskpath
        self.casenumber = CN
        self.SN = SN

    
class DatasetBase:
    
    def __init__(self, train=None, val=None, test=None):
        self.train = train # labeled training data
        self.val = val # validation data (optional)
        self.test = test # test data

    def get_num_classes(self, data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1
    
    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, each_labels_required=2, model=None):
        print(f'Creating a {num_shots}-shot dataset')
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources
        output = []
        for data_source in data_sources:
            tracker = self.split_dataset_by_case(data_source)
            print('tracker: ', tracker.keys())
            for case, items in tracker.items():  # each tracker represents a case, items represent a CT image in the case
                # output.extend(self.get_casedata_by_label(items,num_shots, repeat, each_labels_required=each_labels_required))
                
                num_in_each_group = len(items) // num_shots # directly abandon the few remaining images
                print(num_in_each_group)
                # find the best of each group
                for gp in range(num_shots):
                    features_of_image = []                
                    for i in range(num_in_each_group):
                        image_index = num_in_each_group * gp + i
                        features_of_image.append(self.extract_features(model, items[image_index]))
                        
                    features_of_image = np.array(features_of_image)
                    
                    # brute force calculate the sum of loss
                    sum_of_loss = []
                    for i in range(num_in_each_group):
                        s = 0
                        for j in range(num_in_each_group):
                            if i == j:
                                continue
                            else:
                                s += np.mean((features_of_image[i] - features_of_image[j]) ** 2)         
                        sum_of_loss.append(s)

                    selected_data_index = num_in_each_group * gp + np.argmax(np.array(sum_of_loss))
                    output.append(items[selected_data_index])
                print('cache size: ', len(output))
        return output    
    
    def extract_features(self, model, item):
        img = np.load(item.impath)['image']
        img = cv2.resize(img, (model.img_size, model.img_size), interpolation=cv2.INTER_CUBIC)
        img = (img / 255.0).astype(np.float32)
        img = torch.as_tensor(img)
        if img.dim() == 2: # for gray image
            img = img.unsqueeze(0)
            img = repeat(img, 'c h w -> (repeat c) h w', repeat=3)
        else: # RGB image
            img = img.permute(2, 0, 1)

        with torch.no_grad():
            img_embeddings = model.sam.image_encoder(img.unsqueeze(0).to(model.device)).detach().cpu().numpy()
        return img_embeddings

    def get_casedata_by_label(self, data_source,num_shots,repeat=False,each_labels_required=2):
        selected_data = []
        data_copy = copy.deepcopy(data_source)
        label_count = {key: 0 for key in data_copy[0].label.keys()}
        while True:
            remains = num_shots - len(selected_data)
            if len(data_copy) <= remains: 
                break
            data = random.choice(data_copy)
            
            for key,value in data.label.items():
                if value != "" and label_count[key] < each_labels_required :
                    # it means need to put in selected_data
                    selected_data.append(data)
                    for label_count_key, label_count_value in data.label.items():
                        if label_count_value != '':
                            label_count[label_count_key] += 1
                    break
                            
            data_copy.remove(data)  
            # 每次取個 data 不管適不適合放到cache上都要丟掉，不然有可能會一直 sample 到重複的 data
            
            valid = True  # valid 最後再一起檢查是否全部的label皆已達標
            for key, value in label_count.items():
                if value < each_labels_required:
                    valid = False
                    break

            if valid or (len(data_copy) <= num_shots - len(selected_data)):
                sampled_items = random.sample(data_copy, num_shots - len(selected_data))
                selected_data.extend(sampled_items)
                break
            else:
                continue
        
        return selected_data
        
    def split_dataset_by_case(self, data_source):
        output = defaultdict(list)

        for item in data_source:
            output[item.casenumber].append(item)

        return output

class CacheWrapper(TorchDataset):
    def __init__(self, data_source, desired_size=1024):
        self.data_source = data_source
        self.desired_size = desired_size
        print(f'cahce img size: {desired_size}')
        self.resize = transforms.Resize([desired_size,desired_size], antialias=True)
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1,1,1)
        self.pixel_std = torch.Tensor([53.395, 57.12, 57.375]).view(-1,1,1)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        img = read_image(item.impath)
        img = self.data_transform(img)
        if item.SN is None:
            return img, item.impath, item.label, item.casenumber, 0
        else:
            return img, item.impath, item.label, item.casenumber, item.SN
    
    def data_transform(self, image):
        # start = time.time()
        image = torch.as_tensor(image)
        if image.dim() == 2: # for gray image
            image = self.resize(image)
            if torch.max(image)  > 1:
                image = (image/255.0).float()
            image = image.unsqueeze(0)
            image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        else: # RGB image
            image = image.permute(2,0,1)
            image = self.resize(image)
            image = image / 255
            image = image * 255
            image = torch.clamp(image,0,255)
            image = (image - self.pixel_mean)/self.pixel_std
        # print(f"cache data transform time: {time.time() - start}")
        return image


#TODO:
class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, desired_size=1024, isAug=False):
        self.data_source = data_source
        self.desired_size = desired_size
        print(f'data img size: {desired_size}')
        self.isAug = isAug
        self.resize = transforms.Resize([desired_size,desired_size], antialias=True)
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1,1,1)
        self.pixel_std = torch.Tensor([53.395, 57.12, 57.375]).view(-1,1,1)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'impath': item.impath,
            'label':int(item.label),
            'maskpath': item.maskpath,
            'case' :item.casenumber,
            'slice':item.SN
        }
        img0 = read_image(item.impath)
        Valid = item.maskpath.endswith('.npz') or item.maskpath.endswith('.jpg') or item.maskpath.endswith('.png')
        if Valid:
            mask0 = read_image(item.maskpath,isGT=True)
        if self.isAug:
            angle = np.random.uniform(low=-10, high=10)
            img0,mask0 = self.rotate_data(img0,mask0,angle)
        img = self.train_data_transform(img0)
        output['img'] = img
        if Valid:
            mask = self.mask_transform(mask0)
            output['mask'] = mask
        else:
            output['mask'] = 0
            
        if item.SN is None:
            return output['img'],output['label'],output['mask'],int(output['case']),0
        else:
            return output['img'],output['label'],output['mask'],int(output['case']),output['slice']

    def train_data_transform(self, image):
        # start = time.time()
        image = torch.as_tensor(image)
        
        if image.dim() == 2: # for gray image
            image = self.resize(image)
            if torch.max(image)  > 1:
                image = (image/255.0).float()
            image = image.unsqueeze(0)
            image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        else: # RGB image
            image = image.permute(2,0,1)
            image = self.resize(image)
            image = image / 255
            image = image * 255
            image = torch.clamp(image,0,255)
            image = (image - self.pixel_mean)/self.pixel_std
        # print(f"train data transform time: {time.time() - start}")
        return image

    def mask_transform(self, mask):
        mask = cv2.resize(mask, (self.desired_size, self.desired_size), interpolation=cv2.INTER_CUBIC)
        if np.max(mask) > 1:
            mask = (mask/255.0).astype(np.float32)
        return mask > 0.5
    
    def rotate_data(self,image, mask, angle):
        rows, cols,  = image.shape[:2] 
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_img1 = cv2.warpAffine(image, M, (cols, rows))
        rotated_img2 = cv2.warpAffine(mask, M, (cols, rows))
        return rotated_img1,np.round(rotated_img2)

def build_cache_loader(data_source=None, batch_size=64, shuffle=False, desired_size=1024):
    data_loader = torch.utils.data.DataLoader(
        CacheWrapper(data_source=data_source, desired_size=desired_size),
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available())
    )
    assert len(data_loader) > 0
    return data_loader

def build_data_loader(data_source=None, batch_size=64, shuffle=False, desired_size=1024, isAug=False):
    data_loader = torch.utils.data.DataLoader(
        DatasetWrapper(data_source=data_source, desired_size=desired_size, isAug=isAug),
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available())
    )
    assert len(data_loader) > 0
    return data_loader
