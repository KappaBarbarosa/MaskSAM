import torch
from Model.model import *
from Utils.Score import *
from tqdm import tqdm
import numpy as np
from Model.CacheModel_v2 import CacheModel
from datasets import build_dataset
from datasets.utils_origin import build_data_loader, build_cache_loader
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from Utils.plots import plot_result,plot_slice_result
import wandb
import random
import os

def linear_warmup_cosine_lr_scheduler(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            cos_val = 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
            return cos_val

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

def organ_train(data_config, model, criterions, save_path, model_config,img_save_path,val_config=None, device='cuda:0'):
    training_params = model_config['training']
    prompt = model_config['settings']

    criterion_weights=training_params['criterion_weights'] 
    num_epochs = training_params['num_epochs']
    bs=training_params['batch_size']
    is_train_cache = training_params['train_cache']

    USE_LORA = prompt['USE_LORA']
    USE_TEXT_PROMPT = prompt['USE_TEXT_PROMPT']
    USE_MASK_PROMPT = prompt['USE_MASK_PROMPT']
    
    model = model.to(device)

    CM = None
    if USE_MASK_PROMPT:
        cache_dataset = build_dataset(data_config, image_in_cache=None, is_cache= True, model=model)
        train_loader_cache = build_cache_loader(data_source=cache_dataset.train, batch_size=bs, shuffle=True, desired_size=data_config['img_size'])
        CM = CacheModel(model, train_loader_cache, os.path.join(data_config['root_path'],data_config['label_path']), is_train_cache, USE_LORA, save_path, device,tp_path=model_config['tp_ckpt_path'])  
        if is_train_cache:
            model.CM = CM.to(device)
    
    if training_params['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=float(training_params['lr']), weight_decay=float(training_params['weight_decay']))
    elif training_params['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=float(training_params['lr']), weight_decay=float(training_params['weight_decay']), momentum=0.9)
    scheduler = linear_warmup_cosine_lr_scheduler(optimizer, int(training_params['num_epochs']*0.1), training_params['num_epochs'])

    
    if USE_LORA:
        model.add_lora()
        model = model.to(device)
    else:
        for p in model.parameters():
            p.requires_grad=False
    
    #train common layers for all strategies
    for name, p in model.named_parameters():
        if 'prompt' in name:
            p.requires_grad = True
        if 'decoder' in name:
            p.requires_grad = True
        if 'Text_Embedding_Affine' in name:
            p.requires_grad = True
        if 'transpose' in name:
            p.requires_grad = True
        if 'clip' in name.lower():
            p.requires_grad = False
        if  'CM' in name in name:
            p.requires_grad = True

    if USE_MASK_PROMPT:
        dataset = build_dataset(data_config, image_in_cache=CM.get_image_in_cache(), is_cache=False, model=None)
    else:
        dataset = build_dataset(data_config, image_in_cache=None, is_cache=False, model=None)
    
    tr_dataloader = build_data_loader(data_source=dataset.train, batch_size=bs, shuffle=False, desired_size=data_config['img_size'], isAug=False)
    
    if len(dataset.val)==0:
        sample_size = int(len(dataset.test) * 1)
        dataset.val = random.sample(dataset.test, sample_size)
    val_dataloader = build_data_loader(data_source=dataset.val, batch_size=bs, shuffle=False, desired_size=data_config['img_size'])         

    best_DSC = -np.inf

    for epoch in range(num_epochs):
        train_loss = train_per_epoch(
            model=model, optimizer=optimizer, train_dataloader=tr_dataloader, 
            criterions=criterions, criterion_weights=criterion_weights, 
            CM=CM,
            setting = prompt,
            params = training_params,
            d_cfg = data_config,
            img_save_path = img_save_path,
            epoch=epoch, device=device
        )
        if data_config['name'] =='Synapse':
            valid_metric = valid_per_epoch_for_synapase(
                model=model, valid_dataloader=val_dataloader, 
                CM=CM, d_cfg=data_config,val_config=val_config,
                USE_TEXT = USE_TEXT_PROMPT,
                train_cache = is_train_cache,
                img_save_path = img_save_path,
                epoch=epoch, device=device
            )
            valid_metric /= len(val_config)
            performance = np.mean(valid_metric, axis=0)[0]
            mean_hd95 = np.mean(valid_metric, axis=0)[1]
            wandb.log({'3D Valid DSC':performance, '3D Valid HD':mean_hd95, 'epoch':epoch})
        else:
            performance = valid_per_epoch_for_others(model=model, valid_dataloader=val_dataloader, 
                CM=CM, d_cfg=data_config,
                USE_TEXT = USE_TEXT_PROMPT,label_text = data_config['label_names'],train_cache = is_train_cache,
                img_save_path = img_save_path,
                epoch=epoch, device=device)
            wandb.log({'dsc score':performance, 'epoch':epoch})
           
        if performance > best_DSC:
            best_DSC = performance
            # torch.save(model.transpose.state_dict(), './checkpoints/conv_' + save_path)
            torch.save(model.state_dict(), os.path.join(save_path, 'model_ckpt.pth'))
            if USE_LORA:
                model.save_lora_parameters(os.path.join(save_path, 'lora_p.pth'))
            if USE_MASK_PROMPT:
                CM.save_cache_keys(CM.cache_keys)
        scheduler.step()
    print(f'BEST DSC: {best_DSC}')
    return model


def train_per_epoch( model, CM, optimizer, train_dataloader,criterions, criterion_weights, setting,params,d_cfg,img_save_path,epoch, device):
    model.train()
    
    alpha,beta,label_text,num_classes = d_cfg['alpha'],d_cfg['beta'],d_cfg['label_names'],d_cfg['classes']
    label_weight = torch.tensor(d_cfg['label_weight']) 
    USE_LORA,USE_TEXT, = setting['USE_LORA'], setting['USE_TEXT_PROMPT']
    train_cache = params['train_cache']

    bar = tqdm(enumerate(train_dataloader), total = len(train_dataloader)) 
    losses = torch.zeros(num_classes)
    data_num = len(train_dataloader)/num_classes
    for step, batch in bar:

        image, label, GT, case,SN = batch
        text  = [label_text[i] for i in label.numpy()]
        # SAM encoder
        model.transpose.train()
        image = image.to(device)
        if USE_LORA:
            img_embeds = model.sam.image_encoder(image)
        else:
            with torch.no_grad():
                img_embeds = model.sam.image_encoder(image)

        # Mask Prompt Generator ( CM is None )
        if CM is not None:
            if  train_cache :
                model.CM.train()
                mask_prompts = model.CM(img_embeds=img_embeds, label=label, case=case, beta=beta, is_train=True)
            else:
                with torch.no_grad():
                    mask_prompts = CM(img_embeds=img_embeds, label=label, case=case, beta=beta, is_train=True)

        # Prediction
        if (CM is not None) and (USE_TEXT is True):
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha, x_text=text).squeeze(1)
        elif USE_TEXT is True: # only text prompt
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=None, alpha=alpha, x_text=text).squeeze(1)
        else:                  # only mask prompt
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha, x_text=None).squeeze(1)

        # Loss
        if GT.shape[-2:] != predict_mask.shape[-2:]:
            GT = F.interpolate(GT.float().unsqueeze(1), size=predict_mask.shape[-2:], mode='bicubic').squeeze(1) > 0
        seg_loss=0.0
        loss_array= []
        for criterion, weight in zip(criterions, criterion_weights):
            criterion_name, loss = criterion(predict_mask.float(), GT.float().to(device),label,label_weight)
            loss *= weight
            if loss.numel() == 1:
                seg_loss+=loss
                wandb.log({criterion_name: loss, 'epoch':epoch})
            else:
                seg_loss += loss.mean()
                wandb.log({criterion_name: loss.mean(), 'epoch':epoch})
                loss_array.append(loss)

        seg_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        seg_loss = seg_loss.cpu().item()
        bar.set_postfix(Epoch=epoch, Train_Loss=seg_loss, LR=optimizer.param_groups[0]['lr'])
        
        for loss in loss_array:
            loss = loss.cpu().detach()
            if loss.dim() > 2:
                loss = loss.mean(dim=(1, 2))
            losses.scatter_add_(0, label, loss)
        
        # if CM is None:
        #     mp = np.zeros(GT.shape)
        # else :
        #     mp = mask_prompts.cpu().detach().numpy()
        
        # plot_result(step, image, mp,label.cpu().detach().numpy(), predict_mask, sigmoid_predict_mask, 
        #             GT, threshold_mask,case,SN,path = os.path.join(img_save_path,'train/'))
    losses /= data_num
    for i,loss in enumerate(losses):
        wandb.log({label_text[i]: loss, 'epoch':epoch})

@torch.no_grad()
def valid_per_epoch_for_synapase(model, valid_dataloader, CM, d_cfg, USE_TEXT,train_cache,val_config,img_save_path, epoch, device):
    alpha,beta,classes,input_size,label_text = d_cfg['alpha'],d_cfg['beta'],d_cfg['classes'],d_cfg['img_size'],d_cfg['label_names']
    model.eval()

    Prediction_3d={}
    for case,height in val_config.items():
        pred_blank = np.zeros((classes,height,input_size,input_size))
        gt_blank = np.zeros((classes,height,input_size,input_size))
        C = int(case)
        Prediction_3d[C]={}
        Prediction_3d[C]['pred'] = pred_blank
        Prediction_3d[C]['GT'] = gt_blank
    bar = tqdm(enumerate(valid_dataloader), total = len(valid_dataloader))
    for step, batch in bar:
        image, label, GT,case,SN = batch
        text  = [label_text[i] for i in label.numpy()]

        img_embeds = model.sam.image_encoder(image.to(device))

        if  CM is not None:
            if train_cache:
                model.CM.eval()
                mask_prompts = model.CM(img_embeds=img_embeds, label=label, case=case, beta=beta, is_train=False)
            else:
                mask_prompts = CM(img_embeds=img_embeds, label=label, case=case, beta=beta, is_train=False)
        
        if (CM is not None) and USE_TEXT:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha, x_text=text).squeeze(1)
        elif USE_TEXT:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=None, alpha=alpha, x_text=text).squeeze(1)
        else:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha).squeeze(1)

        predict_mask = F.interpolate(predict_mask.unsqueeze(1), size=GT.shape[-2:], mode='bicubic').squeeze(1)
        sigmoid_predict_mask = torch.sigmoid(predict_mask).detach().cpu().numpy()

        threshold_mask = thresholding(sigmoid_predict_mask)
        for pred,lb,gt,c,s in zip(threshold_mask,label, GT.detach().cpu().numpy(), case,SN):
            Prediction_3d[c.item()]['pred'][lb.item()][s.item()] = pred
            Prediction_3d[c.item()]['GT'][lb.item()][s.item()] = gt
            # dsc,hd95 = calculate_metric_perslice(pred,gt)
#             if  (dsc<0.3) or (lb.item() in [3,7]):
#                 Gtxt = "GNoBlank" if np.sum(gt) > 0 else "GBlank" 
#                 Ptxt = "PNoBlank" if np.sum(pred) > 0 else "PBlank" 
#                 print(f'label:{lb.item()}, case:{c.item()}, slice:{s.item()}, slice score:{dsc,hd95}, {Gtxt}_{Ptxt}')
#                 plot_slice_result(step,img,mp,lb.item(),gt,pred,
#                                   path = os.path.join(img_save_path,'valid/'),
#                                   img_name=f'{c.item()}_{s.item()}_{lb.item()}_{Gtxt}_{Ptxt}')
        bar.set_postfix(Epoch=epoch)

    metric_list = 0.0 
    for case, preds in Prediction_3d.items():
        metric_i = []
        print(f'case:{case}')
        for l in range(classes):
            metric_i.append(calculate_metric_percase(preds['pred'][l], preds['GT'][l]))
        print(metric_i)
        print('case %s mean_dice %f mean_hd95 %f' % (case, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list += np.array(metric_i)

    return metric_list
@torch.no_grad()
def valid_per_epoch_for_others(model, valid_dataloader, CM, d_cfg, USE_TEXT,train_cache,img_save_path, epoch, device):
    alpha,beta,label_text = d_cfg['alpha'],d_cfg['beta'],d_cfg['label_names']
    model.eval()
    running_dice = 0
    count = 0
    bar = tqdm(enumerate(valid_dataloader), total = len(valid_dataloader))
    for step, batch in bar:
        image, label, GT,case,SN = batch
        text  = [label_text[i] for i in label.numpy()]
        img_embeds = model.sam.image_encoder(image.to(device))
        
        if  CM is not None:
            if train_cache:
                model.CM.eval()
                mask_prompts = model.CM(img_embeds=img_embeds, label=label, case=case, beta=beta, is_train=False)
            else:
                mask_prompts = CM(img_embeds=img_embeds, label=label, case=case, beta=beta, is_train=False)
        
        if (CM is not None) and USE_TEXT:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha, x_text=text).squeeze(1)
        elif USE_TEXT:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=None, alpha=alpha, x_text=text).squeeze(1)
        else:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha).squeeze(1)
        
        if GT.shape[-2:] != predict_mask.shape[-2:]:
            predict_mask = F.interpolate(predict_mask.unsqueeze(1), size=GT.shape[-2:], mode='bicubic').squeeze(1)

        ri, ru = running_stats(GT.int().to(device),torch.sigmoid(predict_mask) > 0.5)
        score =dice_collated(ri,ru)
        running_dice += score
        count += ri.shape[0]
        bar.set_postfix(Epoch=epoch, Valid_dsc=score)

        # if CM is None:
        #     mp = np.zeros(GT.shape)
        # else :
        #     mp = mask_prompts.cpu().detach().numpy()
        # sigmoid_mask = torch.sigmoid(predict_mask).detach().cpu().numpy()
        # threshold_mask = sigmoid_mask > 0.5
        # plot_result(step, image, mp,label.cpu().detach().numpy(), predict_mask, sigmoid_mask, 
        #             GT, threshold_mask,istrain=False,path = os.path.join(img_save_path,'valid/'),)
    epoch_dice = running_dice / count
    print(f'Val Dice: {epoch_dice:.4f}')   

    return epoch_dice




