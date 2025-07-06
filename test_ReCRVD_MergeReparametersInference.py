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
from models.RViDeformer_MergeReparametersInference_arch import RViDeformer_MergeReparametersInference
from PIL import Image

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--tile', dest='tile', type=int, default=12, help='tile') 
parser.add_argument('--tile_overlap', dest='tile_overlap', type=int, default=2, help='tile_overlap') 
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

isp = torch.load('isp_pytorch/ISP_RViDeNet/ISP_CNN.pth').cuda()

vis_res = False
choose_version = 2
t = 25
num_frame_testing = args.tile
num_frame_overlapping = args.tile_overlap
stride = num_frame_testing - num_frame_overlapping
t_idx_list = list(range(0, t-num_frame_testing, stride)) + [max(0, t-num_frame_testing)]


#v = 'RViDeformer-L_ReCRVD'
#v = 'RViDeformer-L_un_ReCRVD' #Unsupervised
v = 'RViDeformer-M_ReCRVD'
if "-L_" in v:
    model = RViDeformer_MergeReparametersInference(upscale=1,
            img_size=[8, 64, 64],
            window_size=[6, 8, 8],
            depths=[4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2],
            indep_reconsts=[11, 12],
            embed_dims=[84,84,84,84,84,84,84, 108,108, 108,108],
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            spynet_path=None,
            pa_frames=2,
            deformable_groups=12
            ).cuda()
elif "-M_" in v:
    model = RViDeformer_MergeReparametersInference(upscale=1,
            img_size=[8, 64, 64],
            window_size=[6, 8, 8],
            depths=[4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2],
            indep_reconsts=[11, 12],
            embed_dims=[24,24,24,24,24,24,24, 30,30, 30,30],
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            spynet_path=None,
            pa_frames=2,
            deformable_groups=12
            ).cuda()



f = open('test.txt')
scene_ids = []
for line in f.readlines():
    line = line.strip()
    scene_ids.append(line)
f.close()


exp_name = f'test_{v}'


model_trained_state_dict = torch.load(f'./checkpoints/{v}.pth').state_dict()
model_state_dict = model.state_dict()
model_state_dict.update(model_trained_state_dict)
model.load_state_dict(model_state_dict)

model.eval()

save_dir = "results_ReCRVD"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
f = open(f'./{save_dir}/{exp_name}.txt', 'w')

scene_avg_raw_psnr = 0
scene_avg_raw_ssim = 0
scene_avg_srgb_psnr = 0
scene_avg_srgb_ssim = 0
for scene_id in scene_ids:

    #context = f'{scene_id}\n'
    #f.write(context)

    frame_avg_raw_psnr = 0
    frame_avg_raw_ssim = 0
    frame_avg_srgb_psnr = 0
    frame_avg_srgb_ssim = 0

    save_path_raw = f'{save_dir}/{exp_name}_raw/{scene_id}'
    if not os.path.isdir(save_path_raw):
        os.makedirs(save_path_raw)
    save_path_srgb = f'{save_dir}/{exp_name}_sRGB/{scene_id}'
    if not os.path.isdir(save_path_srgb):
        os.makedirs(save_path_srgb)

    test_noisy_list = []
    test_gt_list = []
    for i in range(1,t+1):

        test_gt = cv2.imread(f'../../data/ReCRVD_dataset/wb_scene_clean_postprocessed/{scene_id}/wb_clean_postprocessed_{i}.tiff',-1)
        test_gt = np.expand_dims(test_gt, axis=0)
        test_gt_list.append(test_gt)

        noisy_path = glob.glob('../../data/ReCRVD_dataset/wb_scene_noisy/{}/*'.format(scene_id))[0] + '/wb_noisy_{}_{}.tiff'.format(i, 0)
        test_noisy = cv2.imread(noisy_path, -1)
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

    for i in range(0,t):

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
            print('saving '+save_path_srgb+'/frame{}_denoised.png'.format(i+1))
            cv2.imwrite(save_path_srgb+'/frame{}_denoised.png'.format(i+1), np.uint8(denoised_srgb_frame[0]*255))
            output = test_result*(2**12-1-240)+240
            save_result = Image.fromarray(np.uint16(output))
            save_result.save(save_path_raw+'/frame{}_denoised.tiff'.format(i+1))

        test_srgb_psnr = compare_psnr(np.uint8(gt_srgb_frame[0]*255).astype(np.float32)/255, np.uint8(denoised_srgb_frame[0]*255).astype(np.float32)/255, data_range=1.0)
        test_srgb_ssim = compare_ssim(np.uint8(gt_srgb_frame[0]*255).astype(np.float32)/255, np.uint8(denoised_srgb_frame[0]*255).astype(np.float32)/255, data_range=1.0, multichannel=True)
        print('scene {} frame{} test raw psnr : {}, test raw ssim : {}, test srgb psnr : {}, test srgb ssim : {}'.format(scene_id, i, test_raw_psnr, test_raw_ssim, test_srgb_psnr, test_srgb_ssim))
        # context = 'frame{} raw psnr/ssim: {}/{}, srgb psnr/ssim'.format(i,test_raw_psnr,test_raw_ssim,test_srgb_psnr,test_srgb_ssim) + '\n'
        # f.write(context)
        frame_avg_raw_psnr += test_raw_psnr
        frame_avg_raw_ssim += test_raw_ssim
        frame_avg_srgb_psnr += test_srgb_psnr
        frame_avg_srgb_ssim += test_srgb_ssim

    frame_avg_raw_psnr = frame_avg_raw_psnr/t
    frame_avg_raw_ssim = frame_avg_raw_ssim/t
    frame_avg_srgb_psnr = frame_avg_srgb_psnr/t
    frame_avg_srgb_ssim = frame_avg_srgb_ssim/t
    # context = 'frame average raw psnr:{}, raw ssim:{}, srgb psnr:{}, srgb ssim:{}'.format(frame_avg_raw_psnr,frame_avg_raw_ssim,frame_avg_srgb_psnr,frame_avg_srgb_ssim) + '\n'
    # f.write(context)

    scene_avg_raw_psnr += frame_avg_raw_psnr
    scene_avg_raw_ssim += frame_avg_raw_ssim
    scene_avg_srgb_psnr += frame_avg_srgb_psnr
    scene_avg_srgb_ssim += frame_avg_srgb_ssim

scene_avg_raw_psnr = scene_avg_raw_psnr/30
scene_avg_raw_ssim = scene_avg_raw_ssim/30
scene_avg_srgb_psnr = scene_avg_srgb_psnr/30
scene_avg_srgb_ssim = scene_avg_srgb_ssim/30

context = 'average raw psnr:{}, raw ssim:{}, srgb psnr:{}, srgb ssim:{}'.format(scene_avg_raw_psnr,scene_avg_raw_ssim,scene_avg_srgb_psnr,scene_avg_srgb_ssim) + '\n'
print(context)
f.write(context)
