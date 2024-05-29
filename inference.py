import os
from importlib import import_module

import numpy as np
import pandas as pd
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader
from datasets.Dataset import SynthDataset

import torch.nn as nn

import yaml
from easydict import EasyDict


def rotationMatrixToEulerAngles(R) :
 
    # assert(isRotationMatrix(R))
     
    sy = torch.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = torch.atan2(R[2,1] , R[2,2])
        y = torch.atan2(-R[2,0], sy)
        z = torch.atan2(R[1,0], R[0,0])
    else :
        x = torch.atan2(-R[1,2], R[1,1])
        y = torch.atan2(-R[2,0], sy)
        z = 0
 
    return torch.tensor([x, y, z])


def euler_to_rotation_matrix(roll, pitch, yaw):
    # Calculate trigonometric values
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    
    # Define the rotation matrix elements
    R = torch.tensor([
        [cp*cy,  cp*sy, -sp],
        [sr*sp*cy - cr*sy,  sr*sp*sy + cr*cy,  sr*cp],
        [cr*sp*cy + sr*sy,  cr*sp*sy - sr*cy,  cr*cp]
    ])
    
    return R.T


def test(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # test_transform = A.Compose([
        # A.Normalize(mean=mean_frh1, std=std_frh1, max_pixel_value=1),
    #     ToTensorV2()
    # ])

    proj_R, proj_t = torch.eye(3), torch.zeros((1, 3))
    proj_R, proj_t = proj_R.to(device).float(), proj_t.to(device).float()


    test_dataset = SynthDataset(args.dataset_path, mode='test', view_num=args.view_num)
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=2,
                            drop_last=True)
    
    print('Start prediction!!')

    # -- model
    model1_module = getattr(import_module('models.' + args.model1_name), args.model1_name)
    model_rt = model1_module(n_channel=2)

    model2_module = getattr(import_module('models.' + args.model2_name), args.model2_name)
    model_ph = model2_module(n_channels=1, n_classes=2)

    model3_module = getattr(import_module('models.' + args.model3_name), args.model3_name)
    model_nval = model3_module(n_channels=2)

    # checkpoint
    checkpoint1 = torch.load(args.model1_path, map_location=device)
    checkpoint2 = torch.load(args.model2_path, map_location=device)
    checkpoint3 = torch.load(args.model3_path, map_location=device)

    model_rt.load_state_dict(checkpoint1)
    model_ph.load_state_dict(checkpoint2)
    model_nval.load_state_dict(checkpoint3)

    # device
    model_rt = model_rt.to(device)  
    model_ph = model_ph.to(device)
    model_nval = model_nval.to(device)  

    model_rt.eval()
    model_ph.eval()
    model_nval.eval()

    criterion = nn.MSELoss()
    total_loss, Rt_loss, ph_loss, nval_loss, cnt = 0, 0, 0, 0, 0

    with torch.no_grad():        
        for step, (imgs, exts, scs, nvals, ini_Rt, file_ids) in enumerate(test_loader):  
            _, _, c, h, w = scs.shape
            abs_Rt = torch.zeros((args.batch_size, args.view_num, 6))
            abs_Rt = abs_Rt.to(device)

            nvals = torch.reshape(nvals, (-1, 1))
            scs = torch.reshape(scs, (-1, c, h, w))
            imgs, exts, scs, nvals = imgs.to(device).float(), exts.to(device).float(), scs.to(device).float(), nvals.to(device).float()
            ini_Rt = ini_Rt.to(device).float()

            # inference
            # 1. rel RT
            pred_Rt = model_rt(imgs)
            loss_R = criterion(pred_Rt[:, :, :3], exts[:, :, :3])
            loss_t = criterion(pred_Rt[:, :, 3:], exts[:, :, 3:])
            loss_Rt = loss_R * 100 + loss_t
            # loss_Rt = criterion(pred_Rt, exts)

            # 2. abs RT
            for b in range(args.batch_size):
                Rt_cur = ini_Rt[b] 
                R_cur, t_cur = Rt_cur[:3], Rt_cur[3:] - proj_t

                Rt_rel = pred_Rt[b]
                # rr_cur = rotationMatrixToEulerAngles(R_cur)
                
                abs_Rt[b, 0, :3] = R_cur
                abs_Rt[b, 0, 3:] = t_cur

                R_cur = euler_to_rotation_matrix(R_cur[0], R_cur[1], R_cur[2])
                R_cur = R_cur.to(device).float()

                for i in range(args.view_num - 1):
                    pred_rel_rt = Rt_rel[i]
                    pred_rel_r = pred_rel_rt[:3]
                    pred_rel_t = pred_rel_rt[3:]
                    
                    pred_rel_rr = euler_to_rotation_matrix(pred_rel_r[0], pred_rel_r[1], pred_rel_r[2])     
                    pred_rel_rr = pred_rel_rr.to(device)

                    R_after = torch.matmul(pred_rel_rr, R_cur)
                    t_after = pred_rel_t + t_cur
    
                    rr_after = rotationMatrixToEulerAngles(R_after)

                    abs_Rt[b, i + 1, :3] = rr_after
                    abs_Rt[b, i + 1, 3:] = t_after
                    
                    R_cur = R_after
                    t_cur = t_after

            abs_Rt = torch.reshape(abs_Rt, (-1, 6))
        
            # 3. phase
            imgs_flatten = torch.zeros((args.batch_size * args.view_num, 1, h, w)) 
            for b in range(args.batch_size):
                for v in range(args.view_num - 1):
                    imgs_flatten[b * args.view_num + v] = imgs[b][v][:1]

                imgs_flatten[b * args.view_num + (args.view_num - 1)]= imgs[b][args.view_num - 2][1:]

            imgs_flatten = imgs_flatten.to(device).float()

            pred_scs = model_ph(imgs_flatten)
            loss_ph = criterion(pred_scs, scs)

            # 4. phase jump
            pred_nvals = model_nval(pred_scs, abs_Rt)
            loss_nval = criterion(pred_nvals, nvals)

            Rt_loss += loss_Rt
            ph_loss += loss_ph
            nval_loss += loss_nval
            total_loss += loss_Rt + loss_ph + loss_nval
            cnt += 1

            pred_ph = torch.atan2(pred_scs[:, 0], pred_scs[:, 1])
            pred_ph = torch.squeeze(pred_ph)
            
            abs_Rt = abs_Rt.detach().cpu().numpy()
            pred_ph = pred_ph.detach().cpu().numpy()
            pred_nvals = pred_nvals.detach().cpu().numpy() 

            # -- save outputs
            file_ids = [list(f) for f in zip(*file_ids)]

            # file_id : view_num X Batch (why?)
            for b in range(args.batch_size):
                files = file_ids[b]
                for v in range(args.view_num):
                    file_id = files[v]
                    # scan_id, img_id = file_id.split('_')
                    
                    rt_path = os.path.join(args.pred_path, 'rt', f'{file_id}.txt')
                    ph_path = os.path.join(args.pred_path, 'ph',  f'{file_id}.csv')
                    n_path = os.path.join(args.pred_path, 'n', f'{file_id}.txt')

                    np.savetxt(rt_path, abs_Rt[b * args.view_num + v, :] , delimiter=',')
                    np.savetxt(ph_path, pred_ph[b * args.view_num + v, :, :] , delimiter=',')
                    np.savetxt(n_path, pred_nvals[b * args.view_num + v, :] , delimiter=',')

        avrg_loss = total_loss / cnt
        Rt_loss, ph_loss, nval_loss = Rt_loss / cnt, ph_loss / cnt, nval_loss / cnt
    
    print(f'Inference || Loss: {round(avrg_loss.item(), 4)} || rt Loss: {round(Rt_loss.item(), 4)} || ph Loss: {round(ph_loss.item(), 4)} || n Loss: {round(nval_loss.item(), 4)}')   
    print("End prediction.")
    
    return 

if __name__ == "__main__":
    CONFIG_FILE_NAME = "./config/config.yaml"
    with open(CONFIG_FILE_NAME, "r") as yml_config_file:
        args = yaml.load(yml_config_file, Loader=yaml.FullLoader)
        args = EasyDict(args["test"])

    if not os.path.exists(args.pred_path):
        os.makedirs(args.pred_path)

    test(args)