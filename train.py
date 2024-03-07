import argparse
import yaml
from Utils.Score import *
from Model.model import Mask_SAM
from trainer import organ_train
from datetime import datetime
import wandb
import json
import os
import random

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_config', default='config_tmp.yml',
                        help='data config file path')

    parser.add_argument('--model_config', default='model_baseline.yml',
                        help='model config file path')
    
    parser.add_argument('--valid_config', default='/work/kappa707/Synapse/val_list.json',
                        help='model config file path')

    parser.add_argument('--pretrained_path', default=None,
                        help='pretrained model path')

    parser.add_argument('--save_path', default='checkpoints/temp.pth',
                        help='pretrained model path')
    parser.add_argument('--img_save_path', default='./results',
                        help='pretrained model path')                 
    parser.add_argument('--device', default='cuda:0', help='device to train on')

    args = parser.parse_args()

    return args



def main_train(data_config, model_config,valid_config, pretrained_path, save_path,img_save_path, device='cuda:0'):
    model_config['img_size'] = data_config['data']['img_size']
    training_params = model_config['training']
    model = Mask_SAM(model_config, device, pretrained_path = pretrained_path)

    #training parameters     
    
    criterion = []

    criterion_mapping = {
    'Dice': dice_loss,
    'BDice': binary_dice_loss,
    'focal': binary_focal_loss,
    'monai focal': monai_focal_loss,
    'monai dice': monai_diceloss,
    'BCE': BCE_loss,
    'weighted dice': weighted_dice_loss,
    'weighted BCE': weighted_BCE_loss,
    'weighted focal': weighted_focal_loss
    }

    for key, value in criterion_mapping.items():
        if key in training_params['criterions']:
            criterion.append(value)
    

    if data_config['data']['name'] == 'Synapse':
        model = organ_train(data_config=data_config['data'], 
                            model=model, 
                            criterions=criterion,
                            save_path=save_path,
                            img_save_path = img_save_path,
                            val_config=valid_config,
                            model_config=model_config,
                            device=device)
    if (data_config['data']['name'] == 'kvasir-instrument') | (data_config['data']['name'] == 'CheXdet'):
        model = organ_train(data_config=data_config['data'], 
                            model=model, 
                            criterions=criterion,
                            save_path=save_path,
                            img_save_path = args.img_save_path,
                            model_config=model_config,
                            device=device)
    return model

if __name__ == '__main__':
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.valid_config, 'r') as f:
        valid_config = json.load(f)
    current_time = datetime.now().strftime("%m-%d_%H:%M")
    random.seed(1)
    torch.manual_seed(1)
    wandb.init(
        # set the wandb project where this run will be logged
        entity="citi2023",
        project="Mask Prompt SAM",
        name=current_time,
        # track hyperparameters and run metadata
        config={
            "dataset": data_config['data']['name'],
            "img_size": data_config['data']['img_size'],
            "shots": data_config['data']['shots'],
            "alpha": data_config['data']['alpha'],
            "beta": data_config['data']['beta'],
            "type": data_config['data']['data_type'],
            "opt": model_config['training']['optimizer'],
            "LR": model_config['training']['lr'],
            "BS": model_config['training']['batch_size'],
            "epoch": model_config['training']['num_epochs'],
            "criterions": model_config['training']['criterions'],
            "criterion_weights": model_config['training']['criterion_weights'],
            "train_cache": model_config['training']['train_cache'],
            "USE_TEXT_PROMPT": model_config['settings']['USE_TEXT_PROMPT'],
            "USE_MASK_PROMPT": model_config['settings']['USE_MASK_PROMPT'],
            "USE_LORA": model_config['settings']['USE_LORA']
        }
    )
    print(model_config)
    path = os.path.join(args.save_path, current_time)
    if not os.path.exists(path):
        os.mkdir(path)
    main_train(data_config, model_config,valid_config, args.pretrained_path, os.path.join(args.save_path, current_time),args.img_save_path, device=args.device)
    wandb.finish()