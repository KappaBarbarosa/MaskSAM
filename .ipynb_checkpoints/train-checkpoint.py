import torch
from data_utils import *
from Model.model import *
from utils import *
from tqdm import tqdm
import numpy as np
from Model.CacheModel import CacheModel
from datasets import build_dataset
from datasets.utils import build_data_loader, build_cache_loader
import wandb
from torch.optim.lr_scheduler import StepLR 
from monai.losses import FocalLoss


from plots import plot_result,plot_slice_result
def train_per_epoch(
        model, optimizer, train_dataloader, 
        criterions, criterion_weights, 
        USE_LORA,
        CM, alpha, beta,train_conv, train_cache, 
        USE_TEXT,label_text,img_save_path,
        epoch, device
    ):
    model.train()
    
    losses = []
    DSC = []
    HD = []
    IOU = []
    metric_list= []
    bar = tqdm(enumerate(train_dataloader), total = len(train_dataloader)) 

    # cross entropy loss
    focal_loss = FocalLoss(reduction='mean', gamma=3.5)
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
            if train_conv or train_cache:
                mask_prompts = CM.get_prompt_masks(img_embeds=img_embeds, label=label, case=case, beta=beta, is_train=True)
            else:
                with torch.no_grad():
                    mask_prompts = CM.get_prompt_masks(img_embeds=img_embeds, label=label, case=case, beta=beta, is_train=True)

        # Prediction
        if (CM is not None) and (USE_TEXT is True):
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha, x_text=text).squeeze(1)
        elif USE_TEXT is True: # only text prompt
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=None, alpha=alpha, x_text=text).squeeze(1)
        else:                  # only mask prompt
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha, x_text=None).squeeze(1)

        # Loss
        GT = F.interpolate(GT.float().unsqueeze(1), size=predict_mask.shape[-2:], mode='bicubic').squeeze(1)
        sigmoid_predict_mask = torch.sigmoid(predict_mask).detach().cpu().numpy()
        seg_loss = 0
        for criterion, weight in zip(criterions, criterion_weights):
            criterion_name, loss = criterion(predict_mask, GT.to(device))
            loss *= weight
            seg_loss += loss
            wandb.log({criterion_name: loss})
        # fcl = focal_loss(predict_mask, GT.float().to(device)) 
        # seg_loss += fcl
        # wandb.log({'focal_loss': fcl})

        seg_loss.backward()
        optimizer.step()
        if (CM is not None) and (train_cache):
                CM.cache_optimizer.step()

        # Scoring
        threshold_mask = thresholding(sigmoid_predict_mask)

        optimizer.zero_grad()
        if (CM is not None) and (train_cache):
            CM.cache_optimizer.zero_grad()
        losses.append(seg_loss.cpu().item())
        bar.set_postfix(Epoch=epoch, Train_Loss=seg_loss.cpu().item(), LR=optimizer.param_groups[0]['lr'])

        # Show results
        if CM is None:
            mp = np.zeros(GT.shape)
        else :
            mp = mask_prompts.cpu().detach().numpy()
        plot_result(step, image, mp,label.cpu().detach().numpy(), predict_mask, sigmoid_predict_mask, 
                    GT, threshold_mask,case,SN,path = os.path.join(img_save_path,'train/'))
    return np.mean(losses), np.mean(DSC), np.mean(HD), np.mean(IOU)

@torch.no_grad()
def valid_per_epoch(
        model, valid_dataloader, 
        criterions, criterion_weights, 
        CM, d_cfg, USE_TEXT,label_text,
        val_config,img_save_path, epoch, device
    ):
    alpha,beta,classes,input_size = d_cfg['alpha'],d_cfg['beta'],d_cfg['classes'],d_cfg['img_size']
    model.eval()
    losses = []
    DSC = []
    HD = []
    IOU = []
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
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

        if CM is not None:
            mask_prompts =  CM.get_prompt_masks(img_embeds=img_embeds, label=label,case=case, beta=beta,is_train=False)
        
        if (CM is not None) and USE_TEXT:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha, x_text=text).squeeze(1)
        elif USE_TEXT:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=None, alpha=alpha, x_text=text).squeeze(1)
        else:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha).squeeze(1)

        predict_mask = F.interpolate(predict_mask.unsqueeze(1), size=GT.shape[-2:], mode='bicubic').squeeze(1)
        sigmoid_predict_mask = torch.sigmoid(predict_mask).detach().cpu().numpy()
        seg_loss = 0.0
        for criterion, weight in zip(criterions, criterion_weights):
            criterion_name, loss = criterion(predict_mask, GT.float().to(device))
            loss *= weight
            seg_loss += loss

        losses.append(seg_loss.cpu().item())
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
        bar.set_postfix(Epoch=epoch, Valid_Loss=seg_loss.cpu().item())

    metric_list = 0.0 
    for case, preds in Prediction_3d.items():
        metric_i = []
        print(f'case:{case}')
        for l in range(classes):
            metric_i.append(calculate_metric_percase(preds['pred'][l], preds['GT'][l]))
        print(metric_i)
        print('case %s mean_dice %f mean_hd95 %f' % (case, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list += np.array(metric_i)

    return np.mean(losses), np.mean(DSC), np.mean(HD), np.mean(IOU),metric_list

def valid_per_epoch_(model, valid_dataloader, CM, d_cfg, USE_TEXT, epoch, device,label_text):
    alpha,beta = d_cfg['alpha'],d_cfg['beta']
    model.eval()
    running_dice = 0
    count = 0
    bar = tqdm(enumerate(valid_dataloader), total = len(valid_dataloader))
    for step, batch in bar:
        image, label, GT,case,SN = batch
        text  = [label_text[i] for i in label.numpy()]
        img_embeds = model.sam.image_encoder(image.to(device))
        if CM is not None:
            mask_prompts =  CM.get_prompt_masks(img_embeds=img_embeds, label=label,case=case, beta=beta,is_train=False)
        
        if (CM is not None) and USE_TEXT:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha, x_text=text).squeeze(1)
        elif USE_TEXT:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=None, alpha=alpha, x_text=text).squeeze(1)
        else:
            predict_mask = model(image_embeddings=img_embeds, mask_prompts=mask_prompts.unsqueeze(1), alpha=alpha).squeeze(1)
        predict_mask = F.interpolate(predict_mask.unsqueeze(1), size=GT.shape[-2:], mode='bicubic').squeeze(1)
        threshold_mask = torch.sigmoid(predict_mask) > 0.4
        ri, ru = running_stats(GT.int().to(device),threshold_mask)
        running_dice += dice_collated(ri,ru)
        count += ri.shape[0]
        bar.set_postfix(Epoch=epoch, Valid_dsc=running_dice)
    epoch_dice = running_dice / count
    print(f'Val Dice: {epoch_dice:.4f}')   

    return epoch_dice




def organ_train(data_config, model, optimizer, scheduler, criterions, save_path, model_config,img_save_path,val_config=None, device='cuda:0'):
    training_params = model_config['training']
    prompt = model_config['settings']

    criterion_weights=training_params['criterion_weights']
    num_epochs = training_params['num_epochs']
    bs=training_params['batch_size']
    is_train_conv = training_params['train_convolution']
    is_train_cache = training_params['train_cache']

    USE_LORA = prompt['USE_LORA']
    USE_TEXT_PROMPT = prompt['USE_TEXT_PROMPT']
    USE_MASK_PROMPT = prompt['USE_MASK_PROMPT']
    
    model = model.to(device)

    CM = None
    if USE_MASK_PROMPT:
        cache_dataset = build_dataset(data_config, image_in_cache=None, is_cache= True, model=model)
        train_loader_cache = build_cache_loader(data_source=cache_dataset.train, batch_size=bs, shuffle=False, desired_size=data_config['img_size'])
        CM = CacheModel(model, train_loader_cache, data_config['image_path'], is_train_conv, is_train_cache, USE_LORA, save_path, device)  
        if is_train_cache:
            CM.cache_optimizer = torch.optim.AdamW(CM.cache_keys.parameters(), lr=float(training_params['lr']), eps=1e-4)
            CM.cache_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(CM.cache_optimizer, training_params['num_epochs'] * len(tr_dataloader))
    
    if USE_LORA:
        model.add_lora()
        model = model.to(device)
    else:
        for p in model.parameters():
            p.requires_grad=False
    
    #train common layers for all strategies
    for name, p in model.named_parameters():
        # if 'norm' in name.lower():
        #     p.requires_grad = True
        # if 'pos_embed' in name.lower():
        #     p.requires_grad = True
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

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    if USE_MASK_PROMPT:
        dataset = build_dataset(data_config, image_in_cache=CM.get_image_in_cache(), is_cache=False, model=None)
    else:
        dataset = build_dataset(data_config, image_in_cache=None, is_cache=False, model=None)
    
    tr_dataloader = build_data_loader(data_source=dataset.train, batch_size=bs, shuffle=True, desired_size=data_config['img_size'], isAug=False)
    val_dataloader = build_data_loader(data_source=dataset.val, batch_size=bs, shuffle=False, desired_size=data_config['img_size'])         

    best_DSC = -np.inf

    for epoch in range(num_epochs):
        train_loss, train_DSC, train_HD, train_IOU = train_per_epoch(
            model=model, optimizer=optimizer, train_dataloader=tr_dataloader, 
            criterions=criterions, criterion_weights=criterion_weights, 
            USE_LORA=USE_LORA,
            CM=CM, alpha=data_config['alpha'], beta=data_config['beta'], train_conv=is_train_conv,train_cache=is_train_cache,
            USE_TEXT = USE_TEXT_PROMPT,label_text = data_config['label_names'],
            img_save_path = img_save_path,
            epoch=epoch, device=device
        )
        wandb.log({'Train DSC':train_DSC, 'Train HD':train_HD, 'Train IOU':train_IOU, 'epoch':epoch})
#         valid_loss, valid_DSC, valid_HD, valid_IOU,valid_metric = valid_per_epoch(
#             model=model, valid_dataloader=val_dataloader, 
#             criterions=criterions, criterion_weights=criterion_weights, 
#             CM=CM, d_cfg=data_config,val_config=val_config,
#             USE_TEXT = USE_TEXT_PROMPT,
#             img_save_path = img_save_path,
#             epoch=epoch, device=device
#         )
#         valid_metric /= len(val_config)
#         performance = np.mean(valid_metric, axis=0)[0]
#         mean_hd95 = np.mean(valid_metric, axis=0)[1]
#         wandb.log({'3D Valid DSC':performance, '3D Valid HD':mean_hd95, 'epoch':epoch})
#         wandb.log({'Valid DSC':valid_DSC, 'Valid HD':valid_HD, 'Valid IOU':valid_IOU, 'epoch':epoch})
        performance = valid_per_epoch_(model=model, valid_dataloader=val_dataloader, 
            CM=CM, d_cfg=data_config,
            USE_TEXT = USE_TEXT_PROMPT,label_text = data_config['label_names'],
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
        if USE_MASK_PROMPT and is_train_cache:
                CM.cache_scheduler.step()
    print(f'BEST DSC: {best_DSC}')
    return model
