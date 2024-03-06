import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import zoom
from einops import repeat
import matplotlib.pyplot as plt
import cv2
import copy

random.seed(22)

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    if(type(path) != str):
        return path
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            if path.endswith('.npz'):
                d = np.load(path)
                img = d['image']
            else:
                img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, maskpath="",CN=0):

        self._impath = impath
        self._label = label
        self.maskpath = maskpath
        self._casenumber = CN

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain



class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = '' # the directory where the dataset is stored
    domains = [] # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self.train_x = train_x # labeled training data
        self.val = val # validation data (optional)
        self.test = test # test data

        #self._num_classes = self.get_num_classes(train_x)
        #self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1
    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=True, num_labels=2
    ):

        print(f'Creating a {num_shots}-shot dataset')
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources
        output = []
        
        for data_source in data_sources:
            tracker = self.split_dataset_by_case(data_source)
            print('tracker: ', tracker.keys())
            dataset = []
            for case, items in tracker.items():
                dataset.extend(self.get_casedata_by_label(items,num_shots, repeat, each_labels_required=2))
                # if len(items) >= num_shots:
                #     sampled_items = random.sample(items, num_shots)
                # else:
                #     if repeat:
                #         sampled_items = random.choices(items, k=num_shots)
                #     else:
                #         sampled_items = items
                # dataset.extend(sampled_items)
            output.append(dataset)
        if len(output) == 1:
            return output[0]

        return output

    def get_casedata_by_label(self, data_source,num_shots,repeat=False,each_labels_required=2):
        selected_data = []
        data_copy = copy.deepcopy(data_source)
        label_count = {key: 0 for key in data_copy[0].label.keys()}
        while True:
            remains = num_shots - len(selected_data)
            if remains <=0 or len(data_copy) <= remains: 
                # 如果data_copy剩下還沒被挑的數量已經和remains一樣了 就全部拿去cache了 沒辦法
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

            if valid:
                sampled_items = random.sample(data_copy, num_shots - len(selected_data))
                selected_data.extend(sampled_items)
                break
            else:
                continue
                
        if(len(selected_data) < num_shots):
            if repeat:
                sampled_items = random.choices(data_copy, k=num_shots - len(selected_data))
            else:
                sampled_items = data_copy
            selected_data.extend(sampled_items)
            
        print(f'label count: {label_count}, selected_data counts: {len(selected_data)}')
        
        return selected_data
        
    def split_dataset_by_case(self, data_source):
        output = defaultdict(list)

        for item in data_source:
            output[item._casenumber].append(item)

        return output


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, desired_size=1024):
        self.data_source = data_source
        self.desired_size = desired_size

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'impath': item.impath,
            'label':item.label,
            'maskpath': item.maskpath ,
        }
        img0 = read_image(item.impath)
        Valid = item.maskpath.endswith('.npz') or item.maskpath.endswith('.jpg') or item.maskpath.endswith('.png')
        if(Valid):
            mask0 = read_image(item.maskpath)
        img = self.train_data_transform(img0)
        output['img'] = img
        if(Valid):
            mask = self.mask_transform(mask0)
            output['mask'] = mask
        else:
            output['mask'] = 0

        return output['img'],output['label'],output['mask']

    def train_data_transform(self, image):
        # print('desire size: ', desired_size)
        image = zoom(image, (self.desired_size // image.shape[-1], self.desired_size // image.shape[-1]), order=0) #TODO:
        
        image = (image * 255.0).astype(np.uint8)
        image = cv2.equalizeHist(image)
        image = (image / 255.0).astype(np.float32)
        
        image = torch.as_tensor(image)
        image = image.unsqueeze(0) 
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        return image

    def mask_transform(self, mask):
        # new_size = self.desired_size // 4
        # mask = zoom(mask, (new_size/ mask.shape[0], new_size / mask.shape[1]), order=0)
        return mask

def build_data_loader(
    data_source=None,
    batch_size=64,
    shuffle=False,
    dataset_wrapper=None,
    desired_size=1024
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source, desired_size=desired_size),
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available())
    )
    assert len(data_loader) > 0

    return data_loader
