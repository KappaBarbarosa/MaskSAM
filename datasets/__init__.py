#from .organ import Organ
from .Synapse import Synapse
from .Endovis17 import Endovis17
from .CheXdet import CheXdet
dataset_list = {
                'Synapse':Synapse,
                'Endovis17':Endovis17,
                'CheXdet':CheXdet,
                'kvasir-instrument':CheXdet
               }

def build_dataset(data_config, image_in_cache, is_cache, model):
    return dataset_list[data_config['name']](data_config, image_in_cache, is_cache, model)