import os
from .utils import Datum, DatasetBase, read_json
class Endovis17(DatasetBase):
    def __init__(self,data_config, image_in_cache, is_cache = False, model=None):
        self.label_path = data_config['label_path']
        self.image_path = data_config['image_path']
        self.image_in_cache = image_in_cache
        json_path = data_config['cache_path'] if is_cache is True else data_config['data_path']
        self.split_path = os.path.join(self.label_path, json_path)
        train, val, test = self.read_split(self.split_path, self.image_path)
        if(is_cache):
            train = self.generate_fewshot_dataset(train, num_shots=data_config['shots'], each_labels_required=3, model=model)

        super().__init__(train=train, val=val, test=test)

    def read_split(self, filepath,root):
        def _convert(items):
            out = []
            for impath, dic, maskpath, CN in items:
                item = Datum(
                    impath=os.path.join(root,impath),
                    label=dic,
                    maskpath=os.path.join(root,maskpath),
                    CN=CN
                )
                out.append(item)
            return out
        def _convert_filter(items):
            out = []
            for impath, dic, maskpath, CN in items:
                if impath in self.image_in_cache:
                    continue
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
        train = _convert(split['train']) if self.image_in_cache is None else _convert_filter(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])
        return train, val, test