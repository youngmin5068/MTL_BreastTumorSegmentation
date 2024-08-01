import os
import torch
import random
import numpy as np
from Metrics import *
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler
import matplotlib.pyplot as plt
import logging
import torch.optim as optim
from dataset import tumor_dataset
import albumentations as A
from loss import DiceLoss
from config import *
import pandas as pd
from albumentations.pytorch import ToTensorV2
from gmic_v2 import GMIC
import cv2
import torch.nn.functional as F



def main(gmic_net,device,train_csv,valid_csv):


    # dir_checkpoint = "/data/save_model"
    o_cols = ["Epoch", "DSC", "Sensitivity", "Precision"]
    train_transform = A.Compose([
        A.Resize(512,512),
        A.HorizontalFlip(p=0.5), 
        #A.VerticalFlip(p=0.5), 
        #A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=4, min_height=16, min_width=16, fill_value=0, p=0.5),
        A.ShiftScaleRotate(shift_limit=(-0.1, 0.1), scale_limit=0, rotate_limit=0, p=0.3, border_mode=cv2.BORDER_REPLICATE),
        #A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])

    valid_transform = A.Compose([
        A.Resize(512,512),
        ToTensorV2()
    ])


    train_dataset = tumor_dataset(csv_path=train_csv,
                                    transform = train_transform,
                                )
    val_dataset = tumor_dataset(csv_path=valid_csv,
                          transform = valid_transform,
                         )
    
    train_loader = DataLoader(train_dataset, num_workers=12, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True)
    val_loader =  DataLoader(val_dataset, num_workers=12, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True)
    
    
    logging.basicConfig(level=logging.INFO)

    logging.info(f'''Starting training:
        Epochs:          {EPOCH}
        Batch size:      {BATCH_SIZE}
        Train size:      {len(train_dataset)}
        Tuning size:     {len(val_dataset)}
        Learning rate:   {LEARNING_RATE}        
        Device:          {device}
    ''')

    optimizer = optim.AdamW(gmic_net.parameters(),betas=(0.99,0.999),lr=LEARNING_RATE) # weight_decay : prevent overfitting
    
    diceloss = DiceLoss()
    bceloss = nn.BCELoss()
    
    best_dice = 0.0
    best_epoch = 1
    
    results = []
    alpha=0.6
    for epoch in range(EPOCH):
        gmic_net.train()
        
        i=1
        for imgs, true_masks,true_labels in train_loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_labels = true_labels.to(device=device,dtype=torch.float32).unsqueeze(-1)
            true_masks = true_masks.to(device=device,dtype=torch.float32)

            optimizer.zero_grad()

            y_global,y_local,y_fusion,output = gmic_net(imgs)

            loss1 = diceloss(torch.sigmoid(output),true_masks)
            loss2 = bceloss(y_global,true_labels)
            loss3 = bceloss(y_local,true_labels)
            loss4 = bceloss(y_fusion,true_labels)
            
            #loss4 = (loss2 + loss3) / 2

            loss = loss1 + loss2 + loss3 + loss4

            loss.backward()
            optimizer.step()

            if i*BATCH_SIZE%500 == 0:
                print('epoch : {}, index : {}/{}, seg_loss:{:.4f}, pred_loss :{:.4f},  total loss: {:.4f}'.format(
                                                                                epoch+1, 
                                                                                i*BATCH_SIZE,
                                                                                len(train_dataset),
                                                                                loss1.detach(),
                                                                                loss4.detach(),
                                                                                loss.detach())
                                                                                ) 
            i += 1

        del imgs
        del true_masks
        del true_labels

        with torch.no_grad():
            print("--------------Validation start----------------")
            gmic_net.eval()     

            dice = 0.0
            sensi = 0.0
            preci = 0.0

            for imgs, true_masks,true_labels in val_loader:
                
                imgs = imgs.to(device=device,dtype=torch.float32)
                true_masks = true_masks.to(device=device,dtype=torch.float32)
                true_labels = true_labels.to(device=device,dtype=torch.float32).unsqueeze(-1)

                y_global,y_local,y_fusion,output = gmic_net(imgs)

                mask_pred = torch.sigmoid(output)
                threshold = torch.zeros_like(mask_pred)
                threshold[mask_pred>0.5] = 1.0
                
                dice += dice_score(threshold, true_masks)
                sensi += recall_score(threshold, true_masks)
                preci += precision_score(threshold, true_masks)
            print("dice sum : {:.4f}, length : {}".format(dice,len(val_loader)))

            mean_dice_score = dice/len(val_loader)
            mean_recall_score = sensi/len(val_loader)
            mean_precision_score = preci/len(val_loader)

            print("current dice : {:.4f}, current recall : {:.4f}, current precision : {:.4f}".format(mean_dice_score,mean_recall_score,mean_precision_score))

            results.append([epoch+1,mean_dice_score.item(),mean_recall_score.item(), mean_precision_score.item()])
            df = pd.DataFrame(results,columns=o_cols)
            df.to_csv(f"/data/youngmin/InterpreSegNet_0620.csv",index=False)

 
            if mean_dice_score > best_dice:
                print("UPDATE dice, loss")
                best_epoch = epoch
                best_dice = mean_dice_score

                try:
                    os.mkdir(DIR_CHECKPOINT)
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                
                torch.save(gmic_net.state_dict(), DIR_CHECKPOINT + f'/InterpreSegNet_0620_{epoch+1}.pth') # d-> custom_UNet_V2 /// e -> att swinUNetr  ////f -> custom unet v2
                logging.info(f'Checkpoint {epoch + 1} saved !')

            print("best epoch : {}, best mean : {:.4f}".format(best_epoch+1,best_dice))
    torch.save(gmic_net.state_dict(), DIR_CHECKPOINT + f'/InterpreSegNet_0620_last.pth')




if __name__ == "__main__":

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    set_seed(MODEL_SEED)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    torch.backends.cudnn.benchmark = True
    

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    #breast_net = SwinUNETR(img_size=(512,512),in_channels=1,out_channels=1,spatial_dims=2,feature_size=36)
    parameters = {
        "device_type":"gpu",
        "gpu_number":0,
        "cam_size": (32, 32),
        "K": 6,
        "crop_shape": (128, 128),
        "percent_t":0.005,
        "post_processing_dim": 512,
        "num_classes": 1
    }

    gmic_net = GMIC(parameters)

    gmic_net.to(device=device)


    main(gmic_net=gmic_net,device=device,train_csv="/data/raw/train/train_meta_new.csv",valid_csv="/data/raw/tuning/tuning_meta_new.csv")

    # np.save("/workspace/IITP/task_3D/dir_checkpoint/swinunetr_losses.npy",losses.cpu().numpy())
    # np.save("/workspace/IITP/task_3D/dir_checkpoint/swinunetr_dices.npy",dices.cpu().numpy())