import matplotlib.pyplot as plt   
import cv2
import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
def plot_result(step, image, mask_prompts,label, predict_mask, sigmoid_predict_mask, GT, threshold_mask,case=None,SN=None,istrain=True,score=None, path='./results1/train/'):
    filename = "{:04d}".format(step)
    ct=0
    image = image.cpu().detach().numpy()
    image = np.transpose(image, (0,2, 3, 1))
    for i,lb in enumerate(label): 
        if istrain & (i >1):
            break
        elif (istrain== False) & (i >5): 
            break          
        fig = plt.figure(figsize=(10, 12))
        plt.subplot(3, 2, 1)
        plt.imshow(image[i])
        plt.colorbar()
        plt.title(f'{lb}_image')

        plt.subplot(3, 2, 2)
        plt.imshow(mask_prompts[i])
        plt.colorbar()
        plt.title('mask prompt')

        plt.subplot(3, 2, 3)
        plt.imshow((predict_mask[i]).cpu().detach().numpy())
        plt.colorbar()
        plt.title('predict')

        plt.subplot(3, 2, 4)
        plt.imshow(sigmoid_predict_mask[i])
        plt.colorbar()
        plt.title('sigmoid_predict')

        plt.subplot(3, 2, 5)
        plt.imshow(threshold_mask[i])
        plt.colorbar()
        plt.title('threshold')

        plt.subplot(3, 2, 6)
        plt.imshow(GT[i])
        plt.colorbar()
        plt.title('GT')
        plt.tight_layout()
        text = ''
        if case is not None:
            text = f'{case[i].item()}_{SN[i].item()}_{lb}'
        if score is not None:
            text = score[i]
        plt.savefig(path + str(filename) + f'{i}_{text}.jpg')
        plt.close()
def plot_slice_result(step, img, mask_prompts,label, GT, threshold_mask, path='./results1/valid/',img_name='image'):
    filename = "{:04d}".format(step)+ '_' + img_name
    image = img.cpu().detach().numpy()
    image = np.transpose(image , (1,2, 0))
    fig = plt.figure(figsize=(10, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.colorbar()
    plt.title(f'{label}_image')

    plt.subplot(2, 2, 2)
    plt.imshow(mask_prompts.cpu().detach().numpy())
    plt.colorbar()
    plt.title('mask prompt')
    plt.subplot(2, 2, 3)
    plt.imshow(threshold_mask)
    plt.colorbar()
    plt.title('threshold')

    plt.subplot(2, 2, 4)
    plt.imshow(GT)
    plt.colorbar()
    plt.title('GT')

    plt.tight_layout()
    plt.savefig(path + str(filename) + '.jpg')
    plt.close()



def plot_test_result(step, image, mask_prompts, predict_mask, sigmoid_predict_mask, GT, threshold_mask,labels,id2label,scores):
    filename = "{:04d}".format(step)
    for i,label in enumerate(labels):
        if (i>1) and ((torch.sum(threshold_mask[i])==0) or (torch.sum(GT[i])==0) or (scores[i]<0.5)):
            print(scores[i])
            continue
        fig = plt.figure(figsize=(10, 12))
        plt.subplot(3, 2, 1)
        plt.imshow(image[i][0].cpu().detach().numpy())
        plt.colorbar()
        plt.title(f'{id2label[label.item()]}__image')

        plt.subplot(3, 2, 2)
        plt.imshow(mask_prompts[i].cpu().detach().numpy())
        plt.colorbar()
        plt.title('mask prompt')


        plt.subplot(3, 2, 3)
        plt.imshow((predict_mask[i]).cpu().detach().numpy())
        plt.colorbar()
        plt.title('predict')

        plt.subplot(3, 2, 4)
        plt.imshow(sigmoid_predict_mask[i])
        plt.colorbar()
        plt.title('sigmoid_predict')

        plt.subplot(3, 2, 5)
        plt.imshow(threshold_mask[i])
        plt.colorbar()
        plt.title('threshold')

        plt.subplot(3, 2, 6)
        plt.imshow(GT[i])
        plt.colorbar()
        plt.title('GT')

        plt.tight_layout()  # 自動調整子圖的佈局
        plt.savefig('./test_results/' + str(filename)+f'_{i}_{scores[i]}' + '.jpg')
        plt.close()
def plot_test_result_v2(filename, image, mask_prompts, predict_mask, sigmoid_predict_mask, GT, threshold_mask,label):
    fig = plt.figure(figsize=(10, 12))
    plt.subplot(3, 2, 1)
    plt.imshow(image)
    plt.colorbar()
    
    plt.title(f'{label}__image')
    plt.subplot(3, 2, 2)
    plt.imshow(mask_prompts.cpu().detach().numpy())
    plt.colorbar()
    plt.title('mask prompt')


    plt.subplot(3, 2, 3)
    plt.imshow((predict_mask).cpu().detach().numpy())
    plt.colorbar()
    plt.title('predict')

    plt.subplot(3, 2, 4)
    plt.imshow(sigmoid_predict_mask)
    plt.colorbar()
    plt.title('sigmoid_predict')

    plt.subplot(3, 2, 5)
    plt.imshow(threshold_mask)
    plt.colorbar()
    plt.title('threshold')

    plt.subplot(3, 2, 6)
    plt.imshow(GT)
    plt.colorbar()
    plt.title('GT')
    
    plt.tight_layout()  # 自動調整子圖的佈局
    plt.savefig('./' + str(filename) + '.jpg')
    plt.close()