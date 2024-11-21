from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.0, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.depth_min = 200
        self.pitch = 36
        self.depth_scale = 200 / ndepths

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    # for light_idx in range(7):
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

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

    def read_phase(self, filename):
        # phase = np.load(filename) / np.pi
        # scale 0~255 to 0~1
        phase = (np.load(filename) + np.pi) / (2 * np.pi)
        phase = phase.astype(np.float32)
        phase = np.expand_dims(phase, axis=0)
 
        return phase
    
    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img
    
    def generate_augmented_data(phase, A_range=(0, 1), B_range=(0, 1)):
        A = np.random.uniform(*A_range)
        B = np.random.uniform(*B_range)
        return A + B * np.cos(phase)

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)
    

    def disp_to_depth(self, disp, min_depth, max_depth):
        min_disp = 1 / max_depth

        max_disp = 1 / min_depth

        scaled_disp = min_disp + (max_disp - min_disp) * disp

        # scaled_disp = scaled_disp.clamp(min=1e-4)
        scaled_disp = np.clip(scaled_disp, 1e-4, None)
        depth = 1 / scaled_disp
        return scaled_disp, depth


    def depth_to_disp(self, depth, min_depth, max_depth):
        scaled_disp = 1 / depth

        min_disp = 1 / max_depth

        max_disp = 1 / min_depth

        disp = (scaled_disp - min_disp) / ((max_disp - min_disp))

        return disp


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        phases = []
        mask = None
        uph = None
        depth_values = None
        proj_matrices = []

        # Random A and B
        # A = np.random.uniform(100, 120)
        # B = np.random.uniform(80, 100)
        A, B = 127.5, 127.5
        gamma = np.random.uniform(0.9, 1.1)  # gamma

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            phase_filename = os.path.join(self.datapath,
                                        'ph_32/{}_train/{:0>3}.npy'.format(scan, vid))
            fringe_path = os.path.join(self.datapath, 'fringe_32/{}_train/{:0>3}.png'.format(scan, vid))
            mask_filename = os.path.join(self.datapath, 'mask/{}_train/{:0>3}.png'.format(scan, vid))
            uph_filename = os.path.join(self.datapath, 'uph_32/{}_train/{:0>3}.npy'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'cam/train/{:0>8}_cam.txt').format(vid)

            ## before
            # phase = self.read_phase(phase_filename)
            # phases.append(phase)

            ## after
            fringe = (cv2.imread(fringe_path, 0) - 127.5) / 127.5
            fringe = fringe.astype(np.float32)
            fringe = np.expand_dims(fringe, axis=0)
            augmented_fringe = (A + B * fringe) / 255
            ## gamma
            # augmented_fringe = np.power(augmented_fringe, gamma) / 255
            phases.append(augmented_fringe)

            intrinsics, extrinsics, _, _ = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                mask = cv2.imread(mask_filename, 0) / 255
                mask = cv2.resize(mask, (160, 128), interpolation=cv2.INTER_NEAREST)
                
                uph = np.load(uph_filename) + self.depth_min
                uph = cv2.resize(uph, (160, 128), interpolation=cv2.INTER_NEAREST)

                ## original
                depth_values = np.arange(self.depth_min, self.depth_scale * self.ndepths + self.depth_min, self.depth_scale, dtype=np.float32)

                ## disp
                # depth_max = self.depth_scale * self.ndepths + self.depth_min
                # disp_min = 1 / depth_max
                # disp_max = 1 / self.depth_min
                # depth_values = np.linspace(disp_min, disp_max, self.ndepths, dtype=np.float32)

                # uph_disp = self.depth_to_disp(uph, self.depth_min, depth_max)
                # uph = uph_disp
                # uph_scaled_disp = self.disp_to_depth(uph_disp, self.depth_min, depth_max)[0]

        uph = uph.astype(np.float32)
        phases = np.stack(phases)
        proj_matrices = np.stack(proj_matrices)

        return {"imgs": phases,
                "proj_matrices": proj_matrices,
                "depth": uph,
                "depth_values": depth_values,
                "mask": mask,
                "filename": f'{scan}_{view_ids[0]:0>3}'}
