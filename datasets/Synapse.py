import os
from .utils import Datum, DatasetBase, read_json
class Synapse(DatasetBase):
    def __init__(self, data_config, image_in_cache, is_cache = False, model=None):
        self.root_path = data_config['root_path']
        self.image_path = os.path.join(self.root_path,data_config['image_path'])
        self.label_path = os.path.join(self.root_path,data_config['label_path'])
        self.image_in_cache = image_in_cache
        data_path = data_config['data_path'] if data_config['data_type'] == 'all' else data_config['Notalldata_path'] 
        json_path = data_config['cache_path'] if is_cache is True else data_path
        self.split_path = os.path.join(self.root_path, json_path)
        train, val, test = self.read_split(self.image_path,self.label_path,self.split_path )
        if(is_cache):
            train = self.generate_fewshot_dataset(train, num_shots=data_config['shots'], each_labels_required=3, model=model)

        super().__init__(train=train, val=val, test=test)

    def read_split(self, imageroot,labelroot,jsonroot):
        def _convert(items,isTest=False):
            out = []
            for impath, dic, maskpath, CN,SN in items:
                item = Datum(
                    impath=os.path.join(imageroot,impath),
                    label=dic,
                    maskpath=os.path.join(labelroot,maskpath),
                    CN=CN,
                    SN=SN
                )
                out.append(item) 
            return out
        def _convert_filter(items):
            out = []
            for impath, dic, maskpath, CN,SN in items:
                if impath in self.image_in_cache:
                    continue
                item = Datum(
                    impath=os.path.join(imageroot,impath),
                    label=dic,
                    maskpath=os.path.join(labelroot,maskpath),
                    CN=CN,
                    SN=SN
                )
                out.append(item)
            return out
        print(f'Reading split from {jsonroot}')
        split = read_json(jsonroot)
        train = _convert(split['train']) if self.image_in_cache is None else _convert_filter(split['train'])
        val = _convert(split['val'],True)
        test = _convert(split['test'],True)
        return train, val, test