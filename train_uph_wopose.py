from importlib import import_module
from pathlib import Path
import os
import random
import math
from math import pi
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.Dataset_uph import SynthDataset
from loss import create_criterion
from scheduler import create_scheduler

from optimizer import create_optimizer

import wandb
import yaml
from easydict import EasyDict

import matplotlib.pyplot as plt
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def save_model(model_ph, model_uph, saved_dir, file_name):
    # rt_output_path = os.path.join(saved_dir, f'{file_name}_rt.pth')
    ph_output_path = os.path.join(saved_dir, f'{file_name}_ph.pth')
    uph_output_path = os.path.join(saved_dir, f'{file_name}_uph.pth')
    # torch.save(model_rt.state_dict(), rt_output_path)
    torch.save(model_ph.state_dict(), ph_output_path)
    torch.save(model_uph.state_dict(), uph_output_path)


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


def GetWrappedPhase(sc):
    '''
        Input: Numerator, Denominator of wrapped phase (B X 2 X H X W)

        Output: wrapped phase (B X 1 X H X W)
    '''
    sin, cos = sc[:, 0], sc[:, 1]

    nudge = (cos == 0) * 1e-7
    cos = cos + nudge
    ph = -torch.atan2(sin, cos)
    ph = ph.unsqueeze(dim=1)
    return ph


def train(args):
    print(f'Start training...')

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- data_set
    # train_transform = A.Compose([
    #     # A.Normalize(mean=mean_frh1_tr, std=std_frh1_tr, max_pixel_value=1),
    #     ToTensorV2()
    # ])

    # val_transform = A.Compose([
    #     # A.Normalize(mean=mean_frh1_val, std=std_frh1_val, max_pixel_value=1),
    #     ToTensorV2()
    # ])

    train_dataset = SynthDataset(args.dataset_path, mode='train', view_num=args.view_num)
    val_dataset = SynthDataset(args.dataset_path, mode='test', view_num=args.view_num)

    # -- datalodaer
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0,
                            drop_last=True)
                                         
    # -- model
    # model1_module = getattr(import_module('models.' + args.model1_name), args.model1_name)
    # model_rt = model1_module(n_channel=2)

    model2_module = getattr(import_module('models.' + args.model2_name), args.model2_name)
    model_ph = model2_module(n_channels=1, n_classes=2)

    model3_module = getattr(import_module('models.' + args.model3_name), args.model3_name)
    model_uph = model3_module(n_channels=args.view_num)

    if args.resume:
        # checkpoint1 = torch.load(args.model1_path, map_location=device)
        checkpoint2 = torch.load(args.model2_path, map_location=device)
        checkpoint3 = torch.load(args.model3_path, map_location=device)

        # model_rt.load_state_dict(checkpoint1)
        model_ph.load_state_dict(checkpoint2)
        model_uph.load_state_dict(checkpoint3)

    # device
    # model_rt = model_rt.to(device)  
    model_ph = model_ph.to(device)
    model_uph = model_uph.to(device)  

    # -- loss & metric
    criterion = create_criterion(args.criterions)
    criterion_val = criterion
    
    # -- optimizer
    # params = list(model_rt.parameters()) + list(model_ph.parameters()) + list(model_uph.parameters())
    params = list(model_ph.parameters()) + list(model_uph.parameters())
    optimizer = create_optimizer(optimizer_name=args.optimizer, params=params, lr=args.lr, weight_decay=5e-4, momentum=0.9)

    # -- scheduler
    if args.scheduler:
        scheduler = create_scheduler(optimizer, args.scheduler, args.epochs, args.lr)

    # -- train 
    best_loss = np.inf
    val_every = 1
    
    # Grad accumulation
    NUM_ACCUM = args.grad_accum
    optimizer.zero_grad()
    
    # Early Stopping
    PATIENCE = args.patience
    counter = 0

    # fp16
    # scaler = torch.cuda.amp.GradScaler()

    proj_R, proj_t = torch.eye(3), torch.zeros((1, 3))
    proj_R, proj_t = proj_R.to(device).float(), proj_t.to(device).float()

    # start train
    for epoch in range(args.epochs):
        print('\n')
        print(f'Epoch : {epoch + 1}')
        # model_rt.train()
        model_ph.train()
        model_uph.train()

        with tqdm(total=len(train_loader)) as pbar:
            for step, (imgs, exts, scs, uphs, ini_Rt, _) in enumerate(train_loader):  
                _, _, c, h, w = scs.shape
                # abs_Rt = torch.zeros((args.batch_size, args.view_num, 6))
                # abs_Rt = abs_Rt.to(device)

                scs = torch.reshape(scs, (-1, c, h, w))
                imgs, exts, scs, uphs = imgs.to(device).float(), exts.to(device).float(), scs.to(device).float(), uphs.to(device).float()
                ini_Rt = ini_Rt.to(device).float()

                # inference
                # 1. rel RT
                # pred_Rt = model_rt(imgs)
                # loss_R = criterion(pred_Rt[:, :, :3], exts[:, :, :3])
                # loss_t = criterion(pred_Rt[:, :, 3:], exts[:, :, 3:])
                # loss_Rt = loss_R * 100 + loss_t
                # loss_Rt = criterion(pred_Rt, exts)

                # # 2. abs RT
                # for b in range(args.batch_size):
                #     Rt_cur = ini_Rt[b]
                #     R_cur, t_cur = Rt_cur[:3], Rt_cur[3:] - proj_t

                #     Rt_rel = pred_Rt[b]
                #     # rr_cur = rotationMatrixToEulerAngles(R_cur)
                    
                #     abs_Rt[b, 0, :3] = R_cur
                #     abs_Rt[b, 0, 3:] = t_cur

                #     R_cur = euler_to_rotation_matrix(R_cur[0], R_cur[1], R_cur[2])
                #     R_cur = R_cur.to(device).float()

                #     for i in range(args.view_num - 1):
                #         pred_rel_rt = Rt_rel[i]
                #         pred_rel_r = pred_rel_rt[:3]
                #         pred_rel_t = pred_rel_rt[3:]
                        
                #         pred_rel_rr = euler_to_rotation_matrix(pred_rel_r[0], pred_rel_r[1], pred_rel_r[2])    
                #         pred_rel_rr = pred_rel_rr.to(device)

                #         R_after = torch.matmul(pred_rel_rr, R_cur)
                #         t_after = pred_rel_t + t_cur
     
                #         rr_after = rotationMatrixToEulerAngles(R_after)

                #         abs_Rt[b, i + 1, :3] = rr_after
                #         abs_Rt[b, i + 1, 3:] = t_after
                        
                #         R_cur = R_after
                #         t_cur = t_after

                # abs_Rt = torch.reshape(abs_Rt, (-1, 6))
                # 3. phase
                imgs_flatten = torch.zeros((args.batch_size * args.view_num, 1, h, w)) 
                for b in range(args.batch_size):
                    for v in range(args.view_num - 1):
                        imgs_flatten[b * args.view_num + v] = imgs[b][v][:1]
    
                    imgs_flatten[b * args.view_num + (args.view_num - 1)]= imgs[b][args.view_num - 2][1:]

                imgs_flatten = imgs_flatten.to(device).float()

                pred_scs = model_ph(imgs_flatten)
                loss_ph = criterion(pred_scs, scs)

                pred_phs = GetWrappedPhase(pred_scs)
                # pred_phs = torch.atan2(pred_scs[:, 0], pred_scs[:, 1])
                # pred_phs = pred_phs.squeeze()
                pred_phs = pred_phs.reshape(args.batch_size, args.view_num, h, w)

                # pred_phs: Batch X View X H X W
                # abs_RT: torch.zeros((args.batch_size, args.view_num, 6))
                # 이거 두개를 fusion 해야 하는데 어캐할까

                # 4. uph

                # pred_uphs = model_uph(pred_phs, pred_Rt)
                pred_uphs = model_uph(pred_phs)
                loss_uph = criterion(pred_uphs, uphs)

                # loss = loss_Rt * args.criterion_weights[0] + loss_ph * args.criterion_weights[1] + loss_uph * args.criterion_weights[2]
                loss = loss_ph * args.criterion_weights[1] + loss_uph * args.criterion_weights[2]

                loss.backward()

                if step % NUM_ACCUM == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                pbar.update(1)
                
                logging = {
                        'Tr Loss': round(loss.item(), 4), 
                        # 'Tr Rt': round(loss_Rt.item(), 4), 
                        'Tr ph': round(loss_ph.item(), 4), 
                        'Tr uph': round(loss_uph.item(), 4), 
                }

                pbar.set_postfix(logging)

                if (step + 1) % args.log_interval == 0:
                    current_lr = get_lr(optimizer)
                    logging['lr'] = current_lr
                    # wandb
                    if args.log_wandb:
                        wandb.log(logging)
            
        # validation / save best model
        if (epoch + 1) % val_every == 0:
            # avrg_loss, Rt_loss, ph_loss, uph_loss = validation(model_rt, model_ph, model_uph, val_loader, device, criterion_val, epoch, args)
            avrg_loss, ph_loss, uph_loss = validation(model_ph, model_uph, val_loader, device, criterion_val, epoch, args)
            if uph_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = uph_loss
                save_model(model_ph, model_uph, saved_dir, args.experiment_name)
                counter = 0
            else:
                counter += 1

            # wandb
            if args.log_wandb:
                wandb.log(
                    {
                        'Val Loss': avrg_loss,
                        # 'Val Rt Loss': Rt_loss,
                        'Val ph Loss': ph_loss,
                        'Val uph Loss': uph_loss,
                    }
                )

            if (args.early_stopping) and (counter > PATIENCE):
                print('Early Stopping...')
                break
        
        if args.scheduler:
            if args.scheduler == 'ReduceOP':
                scheduler.step(avrg_loss)
            else:
                scheduler.step()

    save_model(model_ph, model_uph, saved_dir, args.experiment_name + '_final')


def validation(model_ph, model_uph, data_loader, device, criterion, epoch, args):
    print(f'Start validation!')
    # model_rt.eval()
    model_ph.eval()
    model_uph.eval()
    total_loss, Rt_loss, ph_loss, uph_loss, cnt = 0, 0, 0, 0, 0

    proj_R, proj_t = torch.eye(3), torch.zeros((1, 3))
    proj_R, proj_t = proj_R.to(device).float(), proj_t.to(device).float()

    with torch.no_grad():        
        for step, (imgs, exts, scs, uphs, ini_Rt, _) in enumerate(data_loader):  
            _, _, c, h, w = scs.shape
            # abs_Rt = torch.zeros((args.batch_size, args.view_num, 6))
            # abs_Rt = abs_Rt.to(device)

            scs = torch.reshape(scs, (-1, c, h, w))
            imgs, exts, scs, uphs = imgs.to(device).float(), exts.to(device).float(), scs.to(device).float(), uphs.to(device).float()
            ini_Rt = ini_Rt.to(device).float()

            # inference
            # 1. rel RT
            # pred_Rt = model_rt(imgs)
            # loss_R = criterion(pred_Rt[:, :, :3], exts[:, :, :3])
            # loss_t = criterion(pred_Rt[:, :, 3:], exts[:, :, 3:])
            # loss_Rt = loss_R * 100 + loss_t
            # loss_Rt = criterion(pred_Rt, exts)

            # 2. abs RT
            # for b in range(args.batch_size):
            #     Rt_cur = ini_Rt[b]
            #     R_cur, t_cur = Rt_cur[:3], Rt_cur[3:] - proj_t

            #     Rt_rel = pred_Rt[b]
            #     # rr_cur = rotationMatrixToEulerAngles(R_cur)
                
            #     abs_Rt[b, 0, :3] = R_cur
            #     abs_Rt[b, 0, 3:] = t_cur

            #     R_cur = euler_to_rotation_matrix(R_cur[0], R_cur[1], R_cur[2])
            #     R_cur = R_cur.to(device).float()

            #     for i in range(args.view_num - 1):
            #         pred_rel_rt = Rt_rel[i]
            #         pred_rel_r = pred_rel_rt[:3]
            #         pred_rel_t = pred_rel_rt[3:]
                    
            #         pred_rel_rr = euler_to_rotation_matrix(pred_rel_r[0], pred_rel_r[1], pred_rel_r[2])     
            #         pred_rel_rr = pred_rel_rr.to(device)

            #         R_after = torch.matmul(pred_rel_rr, R_cur)
            #         t_after = pred_rel_t + t_cur
    
            #         rr_after = rotationMatrixToEulerAngles(R_after)

            #         abs_Rt[b, i + 1, :3] = rr_after
            #         abs_Rt[b, i + 1, 3:] = t_after
                    
            #         R_cur = R_after
            #         t_cur = t_after

            # abs_Rt = torch.reshape(abs_Rt, (-1, 6))
        
            # 3. phase
            imgs_flatten = torch.zeros((args.batch_size * args.view_num, 1, h, w)) 
            for b in range(args.batch_size):
                for v in range(args.view_num - 1):
                    imgs_flatten[b * args.view_num + v] = imgs[b][v][:1]

                imgs_flatten[b * args.view_num + (args.view_num - 1)]= imgs[b][args.view_num - 2][1:]

            imgs_flatten = imgs_flatten.to(device).float()

            pred_scs = model_ph(imgs_flatten)
            loss_ph = criterion(pred_scs, scs)

            pred_phs = GetWrappedPhase(pred_scs)
            # pred_phs = torch.atan2(pred_scs[:, 0], pred_scs[:, 1])
            # pred_phs = pred_phs.squeeze()
            pred_phs = pred_phs.reshape(args.batch_size, args.view_num, h, w)

            # 4. uph

            pred_uphs = model_uph(pred_phs)
            loss_uph = criterion(pred_uphs, uphs)

            # Rt_loss += loss_Rt
            ph_loss += loss_ph
            uph_loss += loss_uph
            # total_loss += loss_Rt * args.criterion_weights[0] + loss_ph * args.criterion_weights[1] + uph_loss * args.criterion_weights[2]
            total_loss += loss_ph * args.criterion_weights[1] + uph_loss * args.criterion_weights[2]
            cnt += 1

        avrg_loss = total_loss / cnt
        # Rt_loss, ph_loss, uph_loss = Rt_loss / cnt, ph_loss / cnt, uph_loss / cnt
        ph_loss, uph_loss = ph_loss / cnt, uph_loss / cnt
        print(f'Validation #{epoch + 1} || Loss: {round(avrg_loss.item(), 4)}')
        
    return avrg_loss, ph_loss, uph_loss


if __name__ == "__main__":
    CONFIG_FILE_NAME = "./config/config.yaml"
    with open(CONFIG_FILE_NAME, "r") as yml_config_file:
        args = yaml.load(yml_config_file, Loader=yaml.FullLoader)
        args = EasyDict(args['train'])

    print(args)
    seed_everything(args.seed)

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    CFG = {
        "epochs" : args.epochs,
        "batch_size" : args.batch_size,
        "learning_rate" : args.lr,
        "seed" : args.seed,
        "optimizer" : args.optimizer,
        "scheduler" : args.scheduler,
        "criterion" : args.criterions,
    }

    if args.log_wandb:
        wandb.init(
            project=args.project, entity=args.entity, name=args.experiment_name, config=CFG,
        )

        # wandb.define_metric("Tr Loss", summary="min")
        # wandb.define_metric("Val Loss", summary="min")

    train(args)

    if args.log_wandb:
        wandb.finish()

