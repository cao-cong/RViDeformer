import os
import time
import torch
import torch.nn as nn
import numpy as np
import glob
import cv2
import argparse
from utils import *
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--tile', dest='tile', type=int, default=8, help='tile') 
parser.add_argument('--tile_overlap', dest='tile_overlap', type=int, default=0, help='tile_overlap') 
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

isp = torch.load('isp_pytorch/ISP_RViDeNet/ISP_CNN.pth').cuda()

vis_res = False
choose_version = 2
t = 8
num_frame_testing = args.tile
num_frame_overlapping = args.tile_overlap
stride = num_frame_testing - num_frame_overlapping
t_idx_list = list(range(0, t-num_frame_testing, stride)) + [max(0, t-num_frame_testing)]
print('t_idx_list:{}'.format(t_idx_list))

#v = 'RViDeformer-L_CRVD'
#v = 'RViDeformer-L_un_CRVD'
#v = 'RViDeformer-M_CRVD'
#v = 'RViDeformer-S_CRVD'
v = 'RViDeformer-T_CRVD'
exp_name = f'test_{v}'
save_dir = "results_CRVD_indoor"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
f = open(f'./{save_dir}/{exp_name}.txt', 'w')

model = torch.load(f'./checkpoints/{v}.pth').cuda() # load model
print(model)

model.eval()

iso_list = [1600,3200,6400,12800,25600]

iso_avg_raw_psnr = 0
iso_avg_raw_ssim = 0
iso_avg_srgb_psnr = 0
iso_avg_srgb_ssim = 0
for iso in iso_list:

    scene_avg_raw_psnr = 0
    scene_avg_raw_ssim = 0
    scene_avg_srgb_psnr = 0
    scene_avg_srgb_ssim = 0
    for scene_id in range(7,11+1):

        context = f'{scene_id}\n'
        #f.write(context)

        frame_avg_raw_psnr = 0
        frame_avg_raw_ssim = 0
        frame_avg_srgb_psnr = 0
        frame_avg_srgb_ssim = 0

        save_path = f'{save_dir}/{exp_name}/scene{scene_id}/{iso}'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        test_noisy_list = []
        test_gt_list = []
        for i in range(1,t+1):
        
            if i==8:
                frame_id = 6
            else:
                frame_id = i
            test_gt = cv2.imread(f'../../data/CRVD_dataset/indoor_raw_gt/scene{scene_id}/ISO{iso}/frame{frame_id}_clean_and_slightly_denoised.tiff',-1)
            test_gt = np.expand_dims(test_gt, axis=0)
            test_gt_list.append(test_gt)

            test_noisy = cv2.imread(f'../../data/CRVD_dataset/indoor_raw_noisy/scene{scene_id}/ISO{iso}/frame{frame_id}_noisy0.tiff', -1)
            test_noisy = pack_gbrg_raw(test_noisy)[np.newaxis, ...] 
            test_noisy_list.append(test_noisy)

        test_gt_all = np.concatenate(test_gt_list, axis=0)
        test_noisy = np.concatenate(test_noisy_list, axis=3)
        input_data_v2 = test_noisy.transpose(0,3,1,2)
        input_data_v2 = input_data_v2.reshape(1,t,4,540,960).transpose(0, 1, 3, 4, 2)


        E = np.zeros((1, t, 540, 960, 4))
        W = np.zeros((1, t, 1, 1, 1))
        for t_idx in t_idx_list:
            print(t_idx)
            input_data_v2_clip = input_data_v2[:, t_idx:t_idx+num_frame_testing, :, :, :]
            out_clip = test_big_size_raw_v4(torch.from_numpy(input_data_v2_clip).cuda(), model, patch_h = 64, patch_w = 64, patch_h_overlap = 8, patch_w_overlap = 8)
            out_clip_mask = np.ones((1, min(num_frame_testing, t), 1, 1, 1))
            E[:, t_idx:t_idx+num_frame_testing, ...] += out_clip
            W[:, t_idx:t_idx+num_frame_testing, ...] += out_clip_mask
        test_result_all = E/W

        for i in range(0,t-1):

            test_gt = test_gt_all[i,:,:]
            test_result = test_result_all[0, i:i+1,:,:,:]
            print(test_result.shape)
            test_result = depack_gbrg_raw(test_result)

            test_raw_psnr = compare_psnr((test_gt.astype(np.float32)-240)/(2**12-1-240),(np.uint16(test_result*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240), data_range=1.0)
            test_raw_ssim = compute_ssim_for_packed_raw((test_gt.astype(np.float32)-240)/(2**12-1-240), (np.uint16(test_result*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240))

            gt_raw_frame = np.expand_dims(pack_gbrg_raw(test_gt), axis=0)
            gt_srgb_frame = isp_on_big_size_raw(gt_raw_frame, isp, patch_h = 256, patch_w = 256, patch_hstride = 64, patch_wstride = 64)
            denoised_raw_frame = np.expand_dims(pack_gbrg_raw(test_result*(2**12-1-240)+240), axis=0)
            denoised_srgb_frame = isp_on_big_size_raw(denoised_raw_frame, isp, patch_h = 256, patch_w = 256, patch_hstride = 64, patch_wstride = 64)
            if vis_res:
                print('saving '+save_path+'/frame{}_denoised.png'.format(i))
                cv2.imwrite(save_path+'/frame{}_denoised.png'.format(i), np.uint8(denoised_srgb_frame[0]*255))

            test_srgb_psnr = compare_psnr(np.uint8(gt_srgb_frame[0]*255).astype(np.float32)/255, np.uint8(denoised_srgb_frame[0]*255).astype(np.float32)/255, data_range=1.0)
            test_srgb_ssim = compare_ssim(np.uint8(gt_srgb_frame[0]*255).astype(np.float32)/255, np.uint8(denoised_srgb_frame[0]*255).astype(np.float32)/255, data_range=1.0, multichannel=True)
            print('scene {} frame{} test raw psnr : {}, test raw ssim : {}, test srgb psnr : {}, test srgb ssim : {}'.format(scene_id, i, test_raw_psnr, test_raw_ssim, test_srgb_psnr, test_srgb_ssim))
            # context = 'frame{} raw psnr/ssim: {}/{}, srgb psnr/ssim'.format(i,test_raw_psnr,test_raw_ssim,test_srgb_psnr,test_srgb_ssim) + '\n'
            # f.write(context)
            frame_avg_raw_psnr += test_raw_psnr
            frame_avg_raw_ssim += test_raw_ssim
            frame_avg_srgb_psnr += test_srgb_psnr
            frame_avg_srgb_ssim += test_srgb_ssim

        frame_avg_raw_psnr = frame_avg_raw_psnr/(t-1)
        frame_avg_raw_ssim = frame_avg_raw_ssim/(t-1)
        frame_avg_srgb_psnr = frame_avg_srgb_psnr/(t-1)
        frame_avg_srgb_ssim = frame_avg_srgb_ssim/(t-1)
        # context = 'frame average raw psnr:{}, raw ssim:{}, srgb psnr:{}, srgb ssim:{}'.format(frame_avg_raw_psnr,frame_avg_raw_ssim,frame_avg_srgb_psnr,frame_avg_srgb_ssim) + '\n'
        # f.write(context)

        scene_avg_raw_psnr += frame_avg_raw_psnr
        scene_avg_raw_ssim += frame_avg_raw_ssim
        scene_avg_srgb_psnr += frame_avg_srgb_psnr
        scene_avg_srgb_ssim += frame_avg_srgb_ssim

    scene_avg_raw_psnr = scene_avg_raw_psnr/5
    scene_avg_raw_ssim = scene_avg_raw_ssim/5
    scene_avg_srgb_psnr = scene_avg_srgb_psnr/5
    scene_avg_srgb_ssim = scene_avg_srgb_ssim/5
    
    context = 'iso {} average raw psnr:{}, raw ssim:{}, srgb psnr:{}, srgb ssim:{}'.format(iso,scene_avg_raw_psnr,scene_avg_raw_ssim,scene_avg_srgb_psnr,scene_avg_srgb_ssim) + '\n'
    print(context)
    f.write(context)

    iso_avg_raw_psnr += scene_avg_raw_psnr
    iso_avg_raw_ssim += scene_avg_raw_ssim
    iso_avg_srgb_psnr += scene_avg_srgb_psnr
    iso_avg_srgb_ssim += scene_avg_srgb_ssim

iso_avg_raw_psnr = iso_avg_raw_psnr/5
iso_avg_raw_ssim = iso_avg_raw_ssim/5
iso_avg_srgb_psnr = iso_avg_srgb_psnr/5
iso_avg_srgb_ssim = iso_avg_srgb_ssim/5

context = 'average raw psnr:{}, raw ssim:{}, srgb psnr:{}, srgb ssim:{}'.format(iso_avg_raw_psnr,iso_avg_raw_ssim,iso_avg_srgb_psnr,iso_avg_srgb_ssim) + '\n'
print(context)
f.write(context)
