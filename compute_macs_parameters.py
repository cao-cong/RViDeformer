from __future__ import division
import os
import time
import torch
import functools
from utils import findLastCheckpoint
from ptflops import get_model_complexity_info

from models.RViDeformer_MergeReparameters_arch import RViDeformer_MergeReparameters

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''#RViDeformer-T
denoiser = RViDeformer_MergeReparameters(upscale=1,
               img_size=[8, 64, 64],
               window_size=[6, 8, 8],
               depths=[1, 1, 1, 1, 1, 1, 1, 1, 1],
               indep_reconsts=[11, 12],
               embed_dims=[24,24,24,24,24,24,24, 24,24],
               num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6],
               spynet_path=None,
               pa_frames=2,
               deformable_groups=12
               ).cuda()'''
'''#RViDeformer-S
denoiser = RViDeformer_MergeReparameters(upscale=1,
               img_size=[8, 64, 64],
               window_size=[6, 8, 8],
               depths=[2, 2, 2, 2, 2, 2, 2, 1, 1, 1],
               indep_reconsts=[11, 12],
               embed_dims=[24,24,24,24,24,24,24, 30,30, 30],
               num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
               spynet_path=None,
               pa_frames=2,
               deformable_groups=12
               ).cuda()'''
'''#RViDeformer-M
denoiser = RViDeformer_MergeReparameters(upscale=1,
               img_size=[8, 64, 64],
               window_size=[6, 8, 8],
               depths=[4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2],
               indep_reconsts=[11, 12],
               embed_dims=[24,24,24,24,24,24,24, 30,30, 30,30],
               num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
               spynet_path=None,
               pa_frames=2,
               deformable_groups=12
               ).cuda()'''
#RViDeformer-L
denoiser = RViDeformer_MergeReparameters(upscale=1,
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

denoiser.eval()

upscale = 1
window_size = 8
height = (64 // window_size) * window_size
width = (64 // window_size) * window_size

macs, params = get_model_complexity_info(denoiser, (3, 4, height, width), as_strings=True, print_per_layer_stat=True)
print(f'denoiser : {macs}, {params}')