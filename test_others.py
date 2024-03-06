import os
import numpy as np
from Model.model import *
from Utils.Score import *
from Utils.plots import *
import torch
import yaml
import argparse
from Model.CacheModel import CacheModel
from datasets import build_dataset
from datasets.utils_v2 import build_data_loader
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_config', default=None,
                        help='data config file path')
    parser.add_argument('--model_config', default=None,
                        help='model config file path')
    parser.add_argument('--pretrained_path', default=None,
                        help='pretrained model path')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='testing batch size')
    parser.add_argument('--device', default='cuda:0', help='device to train on')

    args = parser.parse_args()

    return args

@torch.no_grad()
def test_per_epoch(model, test_dataloader, CM, alpha, beta, USE_TEXT,label_text, device):
    model.eval()
    label_score = {}
    bar = tqdm(enumerate(test_dataloader), total = len(test_dataloader))
    for step, batch in bar:
        image, label, GT, case,_ = batch
        img_embeds = model.sam.image_encoder(image.to(device))
        if CM is not None:
            mask_prompts =  CM(img_embeds=img_embeds,case=case, label=label, beta=beta)
        text  = [label_text[i] for i in label.numpy()]
        if (CM is not None) and USE_TEXT: # using both text and mask
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha, x_text=text).squeeze(1)
        elif USE_TEXT: # using only text
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=None, alpha=alpha, x_text=text).squeeze(1)
        else: # using only mask
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha).squeeze(1)
        predict_mask = F.interpolate(predict_mask.unsqueeze(1), size=GT.shape[-2:], mode='bicubic').squeeze(1)
        sigmoid_predict_mask = torch.sigmoid(predict_mask).detach().cpu()
        threshold_mask = sigmoid_predict_mask > 0.5
        dscs=[]
        for pred, ground_truth, l in zip(threshold_mask, GT.detach().cpu(), label):
            dsc = dice_coef(pred, ground_truth)
            dscs.append(dsc)
            iou = calculate_IOU(pred, ground_truth)
            np.round(iou, decimals=0)
            L = label_text[l.item()]
            if L not in label_score:
                label_score[L] = {}
                label_score[L]['DSC'] = []
                label_score[L]['IOU'] = []

            label_score[L]['DSC'].append(dsc)
            label_score[L]['IOU'].append(iou)
        if CM is None:
            mask_prompts= torch.empty_like(GT)
        plot_test_result(step, image, mask_prompts, predict_mask, sigmoid_predict_mask, GT, threshold_mask,label,label_text,dscs)
    return label_score

def test(data_config, model_config, pretrained_path, batch_size, device='cuda:0'):
    with torch.no_grad():
        # model settings
        model_config['img_size'] = data_config['img_size']
        label_text = data_config['label_names']
        USE_LORA = model_config['settings']['USE_LORA']
        USE_TEXT_PROMPT = model_config['settings']['USE_TEXT_PROMPT']
        USE_MASK_PROMPT = model_config['settings']['USE_MASK_PROMPT']
            
        model = Mask_SAM(model_config, device)
        if USE_LORA:
            model.add_lora()

        if pretrained_path is not None:
            state_dict = torch.load(os.path.join(pretrained_path, 'model_ckpt.pth'),map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            if USE_LORA:
                model.load_lora_parameters(os.path.join(pretrained_path, 'lora_p.pth'))

        model = model.to(device)

        # prepare data
        dataset = build_dataset(data_config, image_in_cache=None,is_cache=False, model=None)
        test_dataloader = build_data_loader(data_source=dataset.train, batch_size=batch_size, shuffle=False, desired_size=data_config['img_size'])
        
        CM = None
        if USE_MASK_PROMPT:
            CM = CacheModel(model=model,train_loader_cache=None, image_path=data_config['image_path'], save_or_load_path=pretrained_path, device=device)                             
    
        # testing
        label_score= test_per_epoch(model, test_dataloader, CM, data_config['alpha'], data_config['beta'], USE_TEXT_PROMPT,label_text, device)
        D=I=0
        for key, value in label_score.items():
            print(len(value['DSC']))
            dsc = np.mean(value['DSC'])
            iou = np.mean(value['IOU'])
            print(f'{key}:')
            print(f'\tDSC : {dsc}\n\tIOU : {iou}')
            D+=dsc
            I+=iou
        print(f'mean dsc:{D/len(label_score)}, mean iou: {I/len(label_score)}')

if __name__ == '__main__':
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    os.makedirs('./test_results', exist_ok=True)

    test(data_config['data'], model_config, args.pretrained_path, args.batch_size, device=args.device)

    
