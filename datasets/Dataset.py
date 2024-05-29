import os
import numpy as np
import torch
from torch.utils.data import Dataset
from math import pi
from PIL import Image
import re
import math


## TODO: dataset을 좀 깔끔하게 넣어야 함 

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R) :
 
    # assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


class SynthDataset(Dataset):
    def __init__(self, dataset_path, mode='train', view_num=6, transform=None, interval_scale=1.06, pitch=36):
        super().__init__()
        self.mode = mode
        self.view_num = view_num
        self.listfile = os.path.join(dataset_path, mode + '.txt')
        self.transform = transform
        self.interval_scale = interval_scale
        self.pitch = pitch
        
        self.dataset_path = dataset_path

        self.metas = self.build_list()
        n_val_path = os.path.join(dataset_path, f'n_val_{pitch}.csv')
        self.n_val_list = np.loadtxt(n_val_path, delimiter=',')


    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = os.path.join(self.dataset_path, 'pair.txt')
            # read the pair file
            with open(pair_file) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas
    

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval


    def norm_img(self, img, A=127.5, B=127.5):
        '''
        Input: numpy 2D array
        Output: normalized 2D array
        '''
        # assert(img.ndim == 3)

        normed_img = (img - A) / B

        return normed_img


    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        # np_img = np.array(img, dtype=np.float32) / 255.

        np_img = np.array(img, dtype=np.float32)
        np_img = np.expand_dims(img, axis=2)
        np_img = self.norm_img(np_img)
        # print(np_img.shape)
        np_img = np.transpose(np_img, (2, 0, 1))
        return np_img

    
    def __getitem__(self, index):
        meta = self.metas[index]
        scan, ref_view, src_view = meta
        n_val_list = self.n_val_list[int(scan[4:])]
        view_ids = [ref_view] + src_view[:self.view_num - 1]

        imgs = []
        scs = []
        n_vals = []
        # mask = None
        ext_matrices = []
        R_before, t_before = None, None
        R_cur, t_cur = None, None

        rot_mat_filename = os.path.join(self.dataset_path, 'cam/train/{:0>8}_cam.txt').format(ref_view)
        _, extrinsics, _, _ = self.read_cam_file(rot_mat_filename)
        
        R_ini, t_ini = extrinsics[:3, :3], extrinsics[:-1, -1]
        R2_ini = rotationMatrixToEulerAngles(R_ini)
        ini_Rt = np.concatenate((R2_ini, t_ini))

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.dataset_path,
                                        'fringe_{}/{}_train/{:0>3}.png'.format(self.pitch, scan, vid))
            
            sin_filename = os.path.join(self.dataset_path, 'sin_{}/{}_train/{:0>3}.csv'.format(self.pitch, scan, vid))
            cos_filename = os.path.join(self.dataset_path, 'cos_{}/{}_train/{:0>3}.csv'.format(self.pitch, scan, vid))
            # mask_filename = os.path.join(self.datapath, 'depth/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            rot_mat_filename = os.path.join(self.dataset_path, 'cam/train/{:0>8}_cam.txt').format(vid)
            
            sin, cos = np.loadtxt(sin_filename, delimiter=','), np.loadtxt(cos_filename, delimiter=',')
            sin, cos = np.expand_dims(sin, axis=0), np.expand_dims(cos, axis=0)
            sc = np.concatenate((sin, cos), axis=0)

            imgs.append(self.read_img(img_filename))
            scs.append(sc)
            n_vals.append(n_val_list[vid])

            _, extrinsics, _, _ = self.read_cam_file(rot_mat_filename)
            
            R_cur, t_cur = extrinsics[:3, :3], extrinsics[:-1, -1]
            if R_before is not None:
                rel_R = np.matmul(R_cur, R_before.T)
                rel_t = t_cur - t_before
                rel_R =rotationMatrixToEulerAngles(rel_R)
                RT = np.concatenate((rel_R, rel_t))
                ext_matrices.append(RT)
                
            R_before, t_before = R_cur, t_cur
            # if i == 0:  # reference view
            #     mask = self.read_img(mask_filename)
            
        #stack per 2 images
        imgs = [np.concatenate((imgs[k],imgs[k+1]),axis = 0) for k in range(len(imgs)-1)]

        # imgs = [torch.from_numpy(img) for img in imgs] ## 이거나

        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs) ## 이거 일듯?

        scs = np.stack(scs, axis=0)
        scs = torch.from_numpy(scs) 

        n_vals = np.stack(n_vals, axis=0)
        n_vals = torch.from_numpy(n_vals)

        ext_matrices = np.stack(ext_matrices, axis=0)

        # ext_matrices = [np.matmul(ext_matrices[k], np.linalg.inv(ext_matrices[k+1])) for k in range(len(ext_matrices)-1)]
        # ext_matrices = torch.from_numpy(ext_matrices)
        # ext_matrices = [torch.from_numpy(ext) for ext in ext_matrices]
        
        file_ids = [f'{scan}_{v:03d}' for v in view_ids]
        # return imgs, ext_matrices, mask
        return imgs, ext_matrices, scs, n_vals, ini_Rt, file_ids


    def __len__(self):
        return len(self.metas)