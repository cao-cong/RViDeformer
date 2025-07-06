import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from skimage.measure import compare_psnr
from torch.utils.tensorboard import SummaryWriter
from models.RViDeformer_arch import RViDeformer
from dataset import ReCRVDdataset
from utils import *

parser = argparse.ArgumentParser(description='train video denoising model')
parser.add_argument('--seed', dest='seed', type=int, default=20, help='random seed')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=12000, help='num_epochs') # set epoch num
parser.add_argument('--patch_size', dest='patch_size', type=int, default=64, help='patch_size')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size') # set batch size
parser.add_argument('--num_frame', dest='num_frame', type=int, default=6, help='num_frame')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

model_name = "RViDeformer-L_ReCRVD"
#model_name = "RViDeformer-M_ReCRVD"
exp_name = f'train_{model_name}_ps{args.patch_size}_bs{args.batch_size}_nF{args.num_frame}' # experiment name
use_scheduler = False 
save_interval = 100
learning_rate_base = 1e-4  # set learning rate
learning_rate = learning_rate_base

if args.seed is not None:
    set_seed(args)

save_dir = f'./checkpoints/{exp_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

ps = args.patch_size  # patch size for training
batch_size = args.batch_size # batch size for training
num_frame = args.num_frame

log_dir = f'./logs/{exp_name}'
writer = SummaryWriter(log_dir)

isp = torch.load('isp_pytorch/ISP_RViDeNet/ISP_CNN.pth').cuda()
for k,v in isp.named_parameters():
    v.requires_grad=False


if "-L_" in model_name:
    denoiser = RViDeformer(upscale=1,
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
elif "-M_" in model_name:
    denoiser = RViDeformer(upscale=1,
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



initial_epoch = findLastCheckpoint(save_dir=save_dir)  
print(f'initial epoch: {initial_epoch}')
if initial_epoch > 0:
    print(f'resuming by loading epoch {initial_epoch:03d}')
    denoiser = torch.load(os.path.join(save_dir, f'model_epoch{initial_epoch:d}.pth'))
    initial_epoch += 1

opt = optim.Adam([{'params': denoiser.parameters(), 'initial_lr': learning_rate}], lr = learning_rate)
if use_scheduler:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs+1, eta_min=1e-5, last_epoch=initial_epoch)
    

dataset = ReCRVDdataset(args.patch_size, args.num_frame)

train_data_length = len(dataset)

train_data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=20, shuffle=True)

if initial_epoch==0:
    step=0
else:
    step = (initial_epoch-1)*int(train_data_length/batch_size)
for epoch in range(initial_epoch, args.num_epochs+1):

    train_data_loader_iter = iter(train_data_loader)

    cnt = 0
    
    if not use_scheduler:
        #if epoch>4000:
        #    learning_rate = 0.5*learning_rate_base
        #if epoch>5000:
        #    learning_rate = 0.2*learning_rate_base             
        #latest
        if epoch>4000:
           learning_rate = 0.5*learning_rate_base
        if epoch>10000:
           learning_rate = 0.2*learning_rate_base
        
        for g in opt.param_groups:
            g['lr'] = learning_rate
    else:
        lr_scheduler.step()
        
    for groups in opt.param_groups:
        print('current lr: ',groups['lr'])
        writer.add_scalar('lr', groups['lr'], step)
        
    for batch_id in range(int(train_data_length/batch_size)):
        
        in_data, gt_raw_data = next(train_data_loader_iter)

        in_data = in_data.cuda().permute(0,3,1,2)
        in_data = in_data.reshape(batch_size,num_frame,4,ps,ps)
        
        gt_data = gt_raw_data.cuda().permute(0,3,1,2)
        gt_data = gt_data.reshape(batch_size,num_frame,4,ps,ps)
        gt_isped = isp(gt_data.reshape(-1,4,ps,ps))

        denoiser.train()
        opt.zero_grad()

        denoised_out_s = denoiser(in_data)
        denoised_out_s_isped = isp(denoised_out_s.reshape(-1,4,ps,ps))

        raw_l1_loss = reduce_mean(denoised_out_s, gt_data)
        srgb_l1_loss = reduce_mean(denoised_out_s_isped, gt_isped.detach())

        loss = raw_l1_loss + 0.5*srgb_l1_loss
        loss.backward()
        opt.step()

        cnt += 1
        step += 1
        writer.add_scalar('loss/loss', loss.item(), step)
        writer.add_scalar('loss/raw_l1_loss', raw_l1_loss.item(), step)
        writer.add_scalar('loss/srgb_l1_loss', srgb_l1_loss.item(), step)
        
        print(f"{exp_name} [lr] {round(learning_rate,6)} [epoch]:{epoch:4d} [iter]:{cnt:2d} loss={loss.cpu().item():.6f}")

    if (epoch%save_interval==0) or epoch==args.num_epochs:
        torch.save(denoiser, os.path.join(save_dir, f'model_epoch{epoch:d}.pth'))
        