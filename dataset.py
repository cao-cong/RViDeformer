import cv2
import glob
import random
import numpy as np
import os
from utils import pack_gbrg_raw, generate_name, generate_noisy_raw

# from torchvision import transforms
from torch.utils.data import Dataset


class ReCRVDdataset(Dataset):
    def __init__(self, patch_size, num_frame):
        super(ReCRVDdataset, self).__init__()
        f = open('train.txt')
        self.scene_ids = [line.strip() for line in f.readlines()]
        f.close()

        self.len = len(self.scene_ids)
        self.ps = patch_size
        self.num_frame = num_frame

    def __getitem__(self, index):
        
        scene_id = self.scene_ids[index]
        frame_id = random.randint(1, 25-self.num_frame+1)

        H = 1080-16*2
        W = 1920-16*2

        xx = random.randint(0, W - self.ps*2) // 2 * 2
        yy = random.randint(0, H - self.ps*2) // 2 * 2

        input_pack_list = []
        gt_pack_list = []
        for shift in range(0, self.num_frame):
            id = np.clip(frame_id + shift, 1, 25)


            gt_raw = cv2.imread(f'../../data/ReCRVD_dataset/wb_scene_clean_postprocessed/{scene_id}/wb_clean_postprocessed_{id}.tiff',-1)
            gt_raw = gt_raw[16:-16,16:-16]
            gt_raw_patch = gt_raw[yy:yy + self.ps*2, xx:xx + self.ps*2]
            gt_raw_pack = pack_gbrg_raw(gt_raw_patch)
            gt_pack_list.append(gt_raw_pack)

            noisy_frame_index_for_other = random.randint(0,9)
            # noisy_frame_index_for_other = np.random.randint(0,9+1)
            noisy_raw = cv2.imread(glob.glob(f'../../data/ReCRVD_dataset/wb_scene_noisy/{scene_id}/*')[0] + f'/wb_noisy_{id}_{noisy_frame_index_for_other}.tiff',-1)
            noisy_raw = noisy_raw[16:-16,16:-16]
            noisy_patch = noisy_raw[yy:yy + self.ps*2, xx:xx + self.ps*2]
            input_pack = pack_gbrg_raw(noisy_patch)
            input_pack_list.append(input_pack)
        
        input_pack_frames = np.concatenate(input_pack_list, axis=2)
        gt_pack_frames = np.concatenate(gt_pack_list, axis=2)

        return input_pack_frames, gt_pack_frames

    def __len__(self):
        return len(self.scene_ids)


class CRVDdataset(Dataset):
    def __init__(self, patch_size, num_frame):
        super(CRVDdataset, self).__init__()

        self.len = 30
        self.ps = patch_size
        self.num_frame = num_frame

        self.iso_list = [1600,3200,6400,12800,25600]

    def __getitem__(self, index):

        scene_ind = np.random.randint(1,6+1)
        frame_ind = np.random.randint(1,2+1)
        noisy_level = np.random.randint(1,5+1)

        input_pack_list = []
        gt_pack_list = []
        H = 1080
        W = 1920

        xx = np.random.randint(0, W - self.ps*2+1)
        while xx%2!=0:
            xx = np.random.randint(0, W - self.ps*2+1)
        yy = np.random.randint(0, H - self.ps*2+1)
        while yy%2!=0:
            yy = np.random.randint(0, H - self.ps*2+1)

        for shift in range(0,self.num_frame):

            gt_raw = cv2.imread('../../data/CRVD_dataset/indoor_raw_gt/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(scene_ind, self.iso_list[noisy_level-1], frame_ind+shift),-1)
            #gt_raw_full = gt_raws[data_id]
            gt_raw_full = gt_raw
            gt_raw_patch = gt_raw_full[yy:yy + self.ps*2, xx:xx + self.ps*2]
            gt_raw_pack = pack_gbrg_raw(gt_raw_patch)

            noisy_frame_index_for_other = np.random.randint(0,9+1)
            noisy_raw = cv2.imread('../../data/CRVD_dataset/indoor_raw_noisy/scene{}/ISO{}/frame{}_noisy{}.tiff'.format(scene_ind, self.iso_list[noisy_level-1], frame_ind+shift, noisy_frame_index_for_other),-1)
            noisy_raw_full = noisy_raw
            noisy_patch = noisy_raw_full[yy:yy + self.ps*2, xx:xx + self.ps*2]
            input_pack = pack_gbrg_raw(noisy_patch)
      
            input_pack_list.append(input_pack)
            gt_pack_list.append(gt_raw_pack)
        
        input_pack_frames = np.concatenate(input_pack_list, axis=2)
        gt_pack_frames = np.concatenate(gt_pack_list, axis=2)

        return input_pack_frames, gt_pack_frames

    def __len__(self):
        return self.len