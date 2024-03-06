#from .organ import Organ
from .organ_text import OrganText
dataset_list = {
                'organ_text':OrganText
                }

def build_dataset(label_path, image_path, num_shots, json_path, isfewshot):
    return OrganText(label_path, image_path, num_shots, json_path, isfewshot)