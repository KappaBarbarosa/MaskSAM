import argparse
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_utils import *
from Model.model import Mask_SAM
from test import *
from train import organ_train
from datetime import datetime
import wandb
from torch.optim.lr_scheduler import LambdaLR
import math
from monai.losses import FocalLoss

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_config', default='config_tmp.yml',
                        help='data config file path')

    parser.add_argument('--model_config', default='model_baseline.yml',
                        help='model config file path')
    
    parser.add_argument('--valid_config', default='val_list.json',
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

def linear_warmup_cosine_lr_scheduler(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            cos_val = 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
            return cos_val

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

def main_train(data_config, model_config,valid_config, pretrained_path, save_path,img_save_path, device='cuda:0'):
    model_config['img_size'] = data_config['data']['img_size']
    
    model = Mask_SAM(model_config, device, pretrained_path = pretrained_path)

    #training parameters     
    training_params = model_config['training']
    if training_params['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=float(training_params['lr']), weight_decay=float(training_params['weight_decay']))
    elif training_params['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=float(training_params['lr']), weight_decay=float(training_params['weight_decay']), momentum=0.9)
    exp_lr_scheduler = linear_warmup_cosine_lr_scheduler(optimizer, int(training_params['num_epochs']*0.1), training_params['num_epochs'])

    criterion = []
    if 'dice' in training_params['criterions']:
        criterion.append(dice_loss)
    if 'focal' in training_params['criterions']:
        criterion.append(focal_loss)
    if 'monai focal' in training_params['criterions']:
        criterion.append(monai_focal_loss)
    if 'weighted CE' in training_params['criterions']:
        criterion.append(weighted_ce_loss)
    if 'BCE' in training_params['criterions']:
        criterion.append(BCE_loss)
    if 'weighted dice' in training_params['criterions']:
        criterion.append(weighted_dice_loss)
    if 'weighted BCE' in training_params['criterions']:
        criterion.append(weighted_BCE_loss)
    

    if data_config['data']['name'] == 'Synapse':
        model = organ_train(data_config=data_config['data'], 
                            model=model, 
                            optimizer=optimizer, 
                            scheduler=exp_lr_scheduler, 
                            criterions=criterion,
                            save_path=save_path,
                            img_save_path = img_save_path,
                            model_config=model_config,
                            val_config=valid_config,
                            device=device)
    if (data_config['data']['name'] == 'Endovis17') | (data_config['data']['name'] == 'CheXdet'):
        model = organ_train(data_config=data_config['data'], 
                            model=model, 
                            optimizer=optimizer, 
                            scheduler=exp_lr_scheduler, 
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
    current_time = datetime.now().strftime("%m-%d_%H:%M")
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
            "opt": model_config['training']['optimizer'],
            "LR": model_config['training']['lr'],
            "BS": model_config['training']['batch_size'],
            "epoch": model_config['training']['num_epochs'],
            "loss": model_config['training']['criterions'],
            "loss weights": model_config['training']['criterion_weights'],
            "train_convolution": model_config['training']['train_convolution'],
            "USE_TEXT_PROMPT": model_config['settings']['USE_TEXT_PROMPT'],
            "USE_MASK_PROMPT": model_config['settings']['USE_MASK_PROMPT'],
            "USE_LORA": model_config['settings']['USE_LORA']
        }
    )
    path = os.path.join(args.save_path, current_time)
    if not os.path.exists(path):
        os.mkdir(path)
    main_train(data_config, model_config, args.pretrained_path, os.path.join(args.save_path, current_time),args.img_save_path, device=args.device)
    wandb.finish()
