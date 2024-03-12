import os
import numpy as np
from Model.model import *
from Utils.Score  import *
import torch
import yaml
import argparse
from Model.CacheModel_v2 import CacheModel
from Datasets import build_dataset
from Datasets.utils import build_data_loader
from tqdm import tqdm
from Utils.plots import plot_test_result

import json
i2l=['spleen','right kidney','left kidney','gallbladder','liver','stomach','aorta','pancreas']
label_score = {}
class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'pancreas'}
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
def test_per_epoch(model, test_dataloader, CM,d_cfg,USE_TEXT,test_config,train_cache, device):
    alpha,beta,classes,input_size = d_cfg['alpha'],d_cfg['beta'],d_cfg['classes'],d_cfg['img_size']
    model.eval()
    IOU = []
    bar = tqdm(enumerate(test_dataloader), total = len(test_dataloader))
    Prediction_3d={}
    print(test_config)
    for case,height in test_config.items():
        pred_blank = np.zeros((classes,height,input_size,input_size))
        gt_blank = np.zeros((classes,height,input_size,input_size))
        C = int(case)
        Prediction_3d[C]={}
        Prediction_3d[C]['pred'] = pred_blank
        Prediction_3d[C]['GT'] = gt_blank
    for step, batch in bar:
        # image, label, GT, case = batch
        image, label, GT, case,SN = batch
        # print('label:', label)
        img_embeds = model.sam.image_encoder(image.to(device))
        if CM is not None:
            if train_cache:
                model.CM.eval()
                mask_prompts =  model.CM(img_embeds=img_embeds,case=case, label=label, beta=beta)
            else:   
                mask_prompts =  CM(img_embeds=img_embeds,case=case, label=label, beta=beta)
        
        text  = [i2l[i] for i in label.numpy()]
        if (CM is not None) and USE_TEXT: # using both text and mask
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha, x_text=text).squeeze(1)
        elif USE_TEXT: # using only text
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=None, alpha=alpha, x_text=text).squeeze(1)
        else: # using only mask
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha).squeeze(1)

        # predict_mask = F.interpolate(predict_mask, (512, 512) , mode="bilinear", align_corners=False).squeeze(1)
        predict_mask = F.interpolate(predict_mask.unsqueeze(1), size=GT.shape[-2:], mode='bicubic').squeeze(1)
        sigmoid_predict_mask = torch.sigmoid(predict_mask).detach().cpu().numpy()
        
        threshold_mask = thresholding(sigmoid_predict_mask)
        
        
        for pred,lb,gt,c,s in zip(threshold_mask,label, GT.detach().cpu().numpy(), case,SN):
            try:
                Prediction_3d[c.item()]['pred'][lb.item()][s.item()] = pred
                Prediction_3d[c.item()]['GT'][lb.item()][s.item()] = gt
            except:
                print(f'label:{lb}')
                print(f'case:{c}')
                print(f'slice:{s}')
            
        for pred, ground_truth, l in zip(threshold_mask, GT.detach().cpu().numpy(), label):
#             dsc, hd = calculate_metric_percase(pred, ground_truth)
            iou = calculate_IOU(pred, ground_truth)
            np.round(iou, decimals=0)
            L = i2l[l.item()]
            if L not in label_score:
                label_score[L] = {}
                label_score[L]['IOU'] = []
            label_score[L]['IOU'].append(iou)
            IOU.append(iou)
            
        if CM is None:
            mp = np.zeros(GT.shape[-2:])
        else :
            mp = mask_prompts[0].cpu().detach().numpy()
#         plot_test_result(case,SN, image, mask_prompts, predict_mask, sigmoid_predict_mask, GT, threshold_mask,label,i2l)
    metric_list = 0.0 
    for case, preds in Prediction_3d.items():
        metric_i = []
        print(f'case:{case}')
        for l in range(classes):
            metric_i.append(calculate_metric_percase(preds['pred'][l], preds['GT'][l]))
        print(metric_i)
        print('case %s mean_dice %f mean_hd95 %f' % (case, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list += np.array(metric_i)
    return  np.mean(IOU),metric_list

def test(data_config, model_config, pretrained_path, batch_size,test_config, device='cuda:0'):
    with torch.no_grad():
        # model settings
        model_config['img_size'] = data_config['img_size']
        USE_LORA = model_config['settings']['USE_LORA']
        USE_TEXT_PROMPT = model_config['settings']['USE_TEXT_PROMPT']
        USE_MASK_PROMPT = model_config['settings']['USE_MASK_PROMPT']
        train_cache = model_config['training']['train_cache']
        
        model = Mask_SAM(model_config, device)
        if USE_LORA:
            model.add_lora()
        
        CM = None
        if USE_MASK_PROMPT:
            CM = CacheModel(model=model,train_loader_cache=None, image_path=data_config['image_path'],train_cache=train_cache, save_or_load_path=pretrained_path, device=device,,tp_path=model_config['tp_ckpt_path'])      
        if train_cache:
            model.CM = CM
        if pretrained_path is not None:
            state_dict = torch.load(os.path.join(pretrained_path, 'model_ckpt.pth'),map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            if USE_LORA:
                model.load_lora_parameters(os.path.join(pretrained_path, 'lora_p.pth'))

        model = model.to(device)

        # prepare data
        dataset = build_dataset(data_config, image_in_cache=None,is_cache=False, model=None)
        test_dataloader = build_data_loader(data_source=dataset.train, batch_size=batch_size, shuffle=False, desired_size=data_config['img_size'])
        
                               
    
        # testing
        test_IOU,metric_list = test_per_epoch(model, test_dataloader, CM, data_config,USE_TEXT_PROMPT, test_config,train_cache,device)
        print('Overall Test IOU: ', test_IOU)
        
        
        metric_list = metric_list / len(test_config)
        for i in range(1, data_config['classes'] + 1):
            try:
                print('Mean class %d name %s mean_dice %f mean_hd95 %f' % (i, class_to_name[i], metric_list[i - 1][0], metric_list[i - 1][1]))
            except:
                print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

if __name__ == '__main__':
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open('./train_list.json', 'r') as f:
        test_config = json.load(f)
    os.makedirs('./test_results', exist_ok=True)

    test(data_config['data'], model_config, args.pretrained_path, args.batch_size,test_config, device=args.device)

    for key, value in label_score.items():

        iou = np.mean(value['IOU'])
        print(f'{key}:')
        print(f'\tIOU : {iou}')
