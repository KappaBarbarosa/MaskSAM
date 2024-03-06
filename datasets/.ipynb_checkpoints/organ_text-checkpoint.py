import os
import math
import random
from collections import defaultdict

import torchvision.transforms as transforms

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader


class OrganText(DatasetBase):

    # dataset_dir = '/home/chen0063/SINICA/SAMed/organ_label_withtext'

    def __init__(self, label_path, image_path, num_shots, json_path, isfewshot):
        self.label_path = label_path
        self.image_path = image_path
        self.split_path = os.path.join(self.label_path, json_path)
        train, val, test = self.read_split(self.split_path, self.image_path)
        if(isfewshot):
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
    @staticmethod
    def read_split(filepath,root):
        def _convert(items):
            out = []
            for impath, dic,maskpath,CN in items:
                item = Datum(
                    impath=os.path.join(root,impath),
                    label=dic,
                    maskpath=os.path.join(root,maskpath),
                    CN=CN
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test