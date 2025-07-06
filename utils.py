import os
import re
import torch
import numpy as np
import glob
import random
from scipy.stats import poisson
from skimage.measure import compare_psnr,compare_ssim
import time
import queue, math
import torch.nn.functional as F

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def pack_gbrg_raw(raw):
    #pack GBRG Bayer raw to 4 channels
    black_level = 240
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[0:H:2, 0:W:2, :]), axis=2)
    return out

def pack_gbrg_raw_tensor(raw):
    #pack GBRG Bayer raw to 4 channels
    img_shape = raw.size()
    H = img_shape[2]
    W = img_shape[3]

    out = torch.cat((raw[:, :, 1:H:2, 0:W:2],
                          raw[:, :, 1:H:2, 1:W:2],
                          raw[:, :, 0:H:2, 1:W:2],
                          raw[:, :, 0:H:2, 0:W:2]), dim=1)
    return out

def depack_gbrg_raw(raw):
    H = raw.shape[1]
    W = raw.shape[2]
    output = np.zeros((H*2,W*2))
    for i in range(H):
        for j in range(W):
            output[2*i,2*j]=raw[0,i,j,3]
            output[2*i,2*j+1]=raw[0,i,j,2]
            output[2*i+1,2*j]=raw[0,i,j,0]
            output[2*i+1,2*j+1]=raw[0,i,j,1]
    return output

def pack_rggb_raw(raw, bl, wl):
    #pack RGGB Bayer raw to 4 channels
    black_level = bl
    white_level = wl
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def depack_rggb_raw(raw):
    H = raw.shape[1]
    W = raw.shape[2]
    output = np.zeros((H*2,W*2))
    for i in range(H):
        for j in range(W):
            output[2*i,2*j]=raw[0,i,j,0]
            output[2*i,2*j+1]=raw[0,i,j,1]
            output[2*i+1,2*j]=raw[0,i,j,3]
            output[2*i+1,2*j+1]=raw[0,i,j,2]
    return output

def generate_noisy_raw(gt_raw, a, b):
    """
    a: sigma_s^2
    b: sigma_r^2
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw-240)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    noisy_img = poisson_noisy_img + gaussian_noise + 240
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    
    return noisy_img

def generate_name(number):
    name = list('000000_raw.tiff')
    num_str = str(number)
    for i in range(len(num_str)):
        name[5-i] = num_str[-(i+1)]
    name = ''.join(name)
    return name

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

def reduce_mean_with_weight(im1, im2, noisy_level_data):
    result = torch.abs(im1 - im2) * noisy_level_data * 0.1
    return result.mean()

def preprocess0(raw):
    if type(raw) is np.ndarray:
        input_full = raw.transpose((0,1,4,2,3))
        input_full = torch.from_numpy(input_full)
        input_full = input_full.cuda()
    else:
        input_full = raw.permute((0,1,4,2,3))
    return input_full

def preprocess1(raw):
    input_full = raw.transpose((0, 3, 1, 2))
    input_full = torch.from_numpy(input_full)
    input_full = input_full.cuda()
    return input_full

def preprocess2(raw):
    if type(raw) is np.ndarray:
        input_full = raw.transpose((0,4,1,2,3))
        input_full = torch.from_numpy(input_full)
        input_full = input_full.cuda()
    else:
        input_full = raw.permute((0,4,1,2,3))
    return input_full

def preprocess3(raw):
    input_full = raw.permute((0,1,4,2,3))

    return input_full

def postprocess(output):
    output = output.cpu()
    output = output.detach().numpy().astype(np.float32)
    output = np.transpose(output, (0, 2, 3, 1))
    output = np.clip(output,0,1)
    return output

def postprocess3(output):
    output = output.cpu()
    output = output.detach().numpy().astype(np.float32)
    output = np.transpose(output, (0, 1, 3, 4, 2))
    output = np.clip(output,0,1)
    return output

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def bayer_preserving_augmentation(raw, aug_mode):
    if aug_mode == 0:  # horizontal flip
        aug_raw = np.flip(raw, axis=1)[:,1:-1]
    elif aug_mode == 1: # vertical flip
        aug_raw = np.flip(raw, axis=0)[1:-1,:]
    else:  # random transpose
        aug_raw = np.transpose(raw, (1, 0))
    return aug_raw

def test_big_size_raw(input_data, denoiser, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):

    H = input_data.shape[2]
    W = input_data.shape[3]
    
    test_result = np.zeros((input_data.shape[0],H,W,4))
    t0 = time.clock()
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_patch = input_data[:,:,h_begin:h_end,w_begin:w_end,:]               
            #test_patch = preprocess2(test_patch)
            test_patch = preprocess0(test_patch)
            with torch.no_grad():
                output_patch = denoiser(test_patch)
            test_patch_result = postprocess(output_patch)
            if w_index == 1:
                #print(test_patch_result.shape)
                #print(test_horizontal_result[:,:,w_begin:w_end,:].shape)
                test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
                test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]
            w_index += 1                   
    
        test_patch = input_data[:,:,h_begin:h_end,-patch_w:,:]         
        #test_patch = preprocess2(test_patch)
        test_patch = preprocess0(test_patch)
        with torch.no_grad():
            output_patch = denoiser(test_patch)
        test_patch_result = postprocess(output_patch)       
        last_range = w_end-(W-patch_w)       
        for i in range(last_range):
            test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1)
        test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:]       

        if h_index == 1:
            test_result[:,h_begin:h_end,:,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:,h_begin+i,:,:] = test_result[:,h_begin+i,:,:]*(patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:,i,:,:]*i/(patch_h_overlap-1)
            test_result[:,h_begin+patch_h_overlap:h_end,:,:] = test_horizontal_result[:,patch_h_overlap:,:,:] 
        h_index += 1

    test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_patch = input_data[:,:,-patch_h:,w_begin:w_end,:]               
        #test_patch = preprocess2(test_patch)
        test_patch = preprocess0(test_patch)
        with torch.no_grad():
            output_patch = denoiser(test_patch)
        test_patch_result = postprocess(output_patch)
        if w_index == 1:
            test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
            test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]   
        w_index += 1

    test_patch = input_data[:,:,-patch_h:,-patch_w:,:]         
    #test_patch = preprocess2(test_patch)
    test_patch = preprocess0(test_patch)
    with torch.no_grad():
        output_patch = denoiser(test_patch)
    test_patch_result = postprocess(output_patch)
    last_range = w_end-(W-patch_w)       
    for i in range(last_range):
        test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1) 
    test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:,H-patch_w+i,:,:] = test_result[:,H-patch_w+i,:,:]*(last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:,i,:,:]*i/(last_last_range-1)
    test_result[:,h_end:,:,:] = test_horizontal_result[:,last_last_range:,:,:]
   
    t1 = time.clock()
    print('Total running time: %s s' % (str(t1 - t0)))

    return test_result

def test_big_size_raw_v3(input_data, denoiser, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):
    # input_data: (1, N=3, H, W, C=4)
    input_data = input_data.contiguous()
    H = input_data.shape[2]
    W = input_data.shape[3]
    test_queue = []
    output_queue = queue.Queue()
    test_result = np.zeros((input_data.shape[0],H,W,4)) # (1, H, W, C=4)
    t0 = time.clock()
    # read patchs
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_queue.append(input_data[:,:,h_begin:h_end,w_begin:w_end,:])
            w_index += 1                   
    
        test_queue.append(input_data[:,:,h_begin:h_end,-patch_w:,:])
        h_index += 1

    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_queue.append(input_data[:,:,-patch_h:,w_begin:w_end,:])                
        w_index += 1

    test_queue.append(input_data[:,:,-patch_h:,-patch_w:,:])
    
    # denoise patchs
    input_batch_size = 4
    with torch.no_grad():
        total_patch_num = len(test_queue)
        for i in range(math.ceil(total_patch_num/input_batch_size)):
            top = (i+1)*input_batch_size
            if top<=total_patch_num:
                input_batch = torch.cat(test_queue[i*input_batch_size:top], dim=0)
            else:
                input_batch = torch.cat(test_queue[i*input_batch_size:], dim=0)
            output = denoiser(preprocess2(input_batch))
            for i2 in range(output.size(0)):
                output_queue.put(output[i2:i2+1, :, :, :])
    
    # splice patchs
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            output_patch = output_queue.get()
            test_patch_result = postprocess(output_patch)
            if w_index == 1:
                test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
                test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]
            w_index += 1                   
    
        output_patch = output_queue.get()
        test_patch_result = postprocess(output_patch)       
        last_range = w_end-(W-patch_w)       
        for i in range(last_range):
            test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1)
        test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:]       

        if h_index == 1:
            test_result[:,h_begin:h_end,:,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:,h_begin+i,:,:] = test_result[:,h_begin+i,:,:]*(patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:,i,:,:]*i/(patch_h_overlap-1)
            test_result[:,h_begin+patch_h_overlap:h_end,:,:] = test_horizontal_result[:,patch_h_overlap:,:,:] 
        h_index += 1

    test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        output_patch = output_queue.get()
        test_patch_result = postprocess(output_patch)
        if w_index == 1:
            test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
            test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]   
        w_index += 1

    output_patch = output_queue.get()
    test_patch_result = postprocess(output_patch)
    last_range = w_end-(W-patch_w)       
    for i in range(last_range):
        test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1) 
    test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:,H-patch_w+i,:,:] = test_result[:,H-patch_w+i,:,:]*(last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:,i,:,:]*i/(last_last_range-1)
    test_result[:,h_end:,:,:] = test_horizontal_result[:,last_last_range:,:,:]
    
    t1 = time.clock()
    print('Total running time: %s s' % (str(t1 - t0)))

    return test_result

def test_big_size_raw_v4(input_data, denoiser, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):
    # input_data: (1, N=3, H, W, C=4)
    input_data = input_data.contiguous()
    H = input_data.shape[2]
    W = input_data.shape[3]
    test_queue = []
    output_queue = queue.Queue()
    #test_result = np.zeros((input_data.shape[0],H,W,4)) # (1, H, W, C=4)
    test_result = np.zeros_like(input_data.detach().cpu().numpy())
    #t0 = time.clock()
    # read patchs
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_queue.append(input_data[:,:,h_begin:h_end,w_begin:w_end,:])
            w_index += 1                   
    
        test_queue.append(input_data[:,:,h_begin:h_end,-patch_w:,:])
        h_index += 1

    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_queue.append(input_data[:,:,-patch_h:,w_begin:w_end,:])                
        w_index += 1

    test_queue.append(input_data[:,:,-patch_h:,-patch_w:,:])
    
    # denoise patchs
    input_batch_size = 8
    with torch.no_grad():
        total_patch_num = len(test_queue)
        for i in range(math.ceil(total_patch_num/input_batch_size)):
            top = (i+1)*input_batch_size
            if top<=total_patch_num:
                input_batch = torch.cat(test_queue[i*input_batch_size:top], dim=0)
            else:
                input_batch = torch.cat(test_queue[i*input_batch_size:], dim=0)
            #print(input_batch.shape)
            #print(preprocess3(input_batch).shape)
            output = denoiser(preprocess3(input_batch))
            for i2 in range(output.size(0)):
                output_queue.put(output[i2:i2+1, :, :, :])
    
    # splice patchs
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = np.zeros((input_data.shape[0],input_data.shape[1],patch_h,W,4))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            output_patch = output_queue.get()
            test_patch_result = postprocess3(output_patch)
            if w_index == 1:
                test_horizontal_result[:,:,:,w_begin:w_end,:] = test_patch_result
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:,:,:,w_begin+i,:] = test_horizontal_result[:,:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,:,i,:]*i/(patch_w_overlap-1)
                test_horizontal_result[:,:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,:,patch_w_overlap:,:]
            w_index += 1                   
    
        output_patch = output_queue.get()
        test_patch_result = postprocess3(output_patch)       
        last_range = w_end-(W-patch_w)       
        for i in range(last_range):
            test_horizontal_result[:,:,:,W-patch_w+i,:] = test_horizontal_result[:,:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,:,i,:]*i/(last_range-1)
        test_horizontal_result[:,:,:,w_end:,:] = test_patch_result[:,:,:,last_range:,:]       

        if h_index == 1:
            test_result[:,:,h_begin:h_end,:,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:,:,h_begin+i,:,:] = test_result[:,:,h_begin+i,:,:]*(patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:,:,i,:,:]*i/(patch_h_overlap-1)
            test_result[:,:,h_begin+patch_h_overlap:h_end,:,:] = test_horizontal_result[:,:,patch_h_overlap:,:,:] 
        h_index += 1

    test_horizontal_result = np.zeros((input_data.shape[0],input_data.shape[1],patch_h,W,4))
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        output_patch = output_queue.get()
        test_patch_result = postprocess3(output_patch)
        if w_index == 1:
            test_horizontal_result[:,:,:,w_begin:w_end,:] = test_patch_result
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:,:,:,w_begin+i,:] = test_horizontal_result[:,:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,:,i,:]*i/(patch_w_overlap-1)
            test_horizontal_result[:,:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,:,patch_w_overlap:,:]   
        w_index += 1

    output_patch = output_queue.get()
    test_patch_result = postprocess3(output_patch)
    last_range = w_end-(W-patch_w)       
    for i in range(last_range):
        test_horizontal_result[:,:,:,W-patch_w+i,:] = test_horizontal_result[:,:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,:,i,:]*i/(last_range-1) 
    test_horizontal_result[:,:,:,w_end:,:] = test_patch_result[:,:,:,last_range:,:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:,:,H-patch_w+i,:,:] = test_result[:,:,H-patch_w+i,:,:]*(last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:,:,i,:,:]*i/(last_last_range-1)
    test_result[:,:,h_end:,:,:] = test_horizontal_result[:,:,last_last_range:,:,:]
    
    #t1 = time.clock()
    #print('Total running time: %s s' % (str(t1 - t0)))

    return test_result

def isp_on_big_size_raw(input_data, isp, patch_h = 256, patch_w = 256, patch_hstride = 64, patch_wstride = 64):

    H = input_data.shape[1]
    W = input_data.shape[2]
    
    test_result = np.zeros((input_data.shape[0],H*2,W*2,3))
    #t0 = time.clock()
    h_index = 1
    while (patch_h*h_index-patch_hstride*(h_index-1)) < H:
        test_horizontal_result = np.zeros((input_data.shape[0],patch_h*2,W*2,3))
        h_begin = patch_h*(h_index-1)-patch_hstride*(h_index-1)
        h_end = patch_h*h_index-patch_hstride*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_wstride*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_wstride*(w_index-1)
            w_end = patch_w*w_index-patch_wstride*(w_index-1)
            test_patch = input_data[:,h_begin:h_end,w_begin:w_end,:]               
            test_patch = preprocess1(test_patch)               
            with torch.no_grad():
                output_patch = isp(test_patch)
            test_patch_result = postprocess(output_patch)
            if w_index == 1:
                test_horizontal_result[:,:,w_begin*2:w_end*2,:] = test_patch_result
            else:
                for i in range(patch_wstride*2):
                    test_horizontal_result[:,:,2*w_begin+i,:] = test_horizontal_result[:,:,2*w_begin+i,:]*(patch_wstride*2-1-i)/(patch_wstride*2-1)+test_patch_result[:,:,i,:]*i/(patch_wstride*2-1)
                test_horizontal_result[:,:,2*w_begin+2*patch_wstride:w_end*2,:] = test_patch_result[:,:,2*patch_wstride:,:]
            w_index += 1                   
    
        test_patch = input_data[:,h_begin:h_end,-patch_w:,:]         
        test_patch = preprocess1(test_patch)
        with torch.no_grad():
            output_patch = isp(test_patch)
        test_patch_result = postprocess(output_patch)       
        last_range = w_end-(W-patch_w)       
        for i in range(last_range*2):
            test_horizontal_result[:,:,W*2-patch_w*2+i,:] = test_horizontal_result[:,:,W*2-patch_w*2+i,:]*(last_range*2-1-i)/(last_range*2-1)+test_patch_result[:,:,i,:]*i/(last_range*2-1)
        test_horizontal_result[:,:,w_end*2:,:] = test_patch_result[:,:,last_range*2:,:]       

        if h_index == 1:
            test_result[:,h_begin*2:h_end*2,:,:] = test_horizontal_result
        else:
            for i in range(patch_hstride*2):
                test_result[:,h_begin*2+i,:,:] = test_result[:,h_begin*2+i,:,:]*(patch_hstride*2-1-i)/(patch_hstride*2-1)+test_horizontal_result[:,i,:,:]*i/(patch_hstride*2-1)
            test_result[:,h_begin*2+patch_hstride*2:h_end*2,:,:] = test_horizontal_result[:,patch_hstride*2:,:,:] 
        h_index += 1

    test_horizontal_result = np.zeros((input_data.shape[0],patch_h*2,W*2,3))
    w_index = 1
    while (patch_w*w_index-patch_wstride*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_wstride*(w_index-1)
        w_end = patch_w*w_index-patch_wstride*(w_index-1)
        test_patch = input_data[:,-patch_h:,w_begin:w_end,:]               
        test_patch = preprocess1(test_patch)               
        with torch.no_grad():
            output_patch = isp(test_patch)
        test_patch_result = postprocess(output_patch)
        if w_index == 1:
            test_horizontal_result[:,:,w_begin*2:w_end*2,:] = test_patch_result
        else:
            for i in range(patch_wstride*2):
                test_horizontal_result[:,:,w_begin*2+i,:] = test_horizontal_result[:,:,w_begin*2+i,:]*(patch_wstride*2-1-i)/(patch_wstride*2-1)+test_patch_result[:,:,i,:]*i/(patch_wstride*2-1)
            test_horizontal_result[:,:,w_begin*2+patch_wstride*2:w_end*2,:] = test_patch_result[:,:,patch_wstride*2:,:]   
        w_index += 1

    test_patch = input_data[:,-patch_h:,-patch_w:,:]         
    test_patch = preprocess1(test_patch)
    with torch.no_grad():
        output_patch = isp(test_patch)
    test_patch_result = postprocess(output_patch)
    last_range = w_end-(W-patch_w)       
    for i in range(last_range*2):
        test_horizontal_result[:,:,W*2-patch_w*2+i,:] = test_horizontal_result[:,:,W*2-patch_w*2+i,:]*(last_range*2-1-i)/(last_range*2-1)+test_patch_result[:,:,i,:]*i/(last_range*2-1) 
    test_horizontal_result[:,:,w_end*2:,:] = test_patch_result[:,:,last_range*2:,:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range*2):
        test_result[:,H*2-patch_w*2+i,:,:] = test_result[:,H*2-patch_w*2+i,:,:]*(last_last_range*2-1-i)/(last_last_range*2-1)+test_horizontal_result[:,i,:,:]*i/(last_last_range*2-1)
    test_result[:,h_end*2:,:,:] = test_horizontal_result[:,last_last_range*2:,:,:]
   
    #t1 = time.clock()
    #print('Total running time: %s s' % (str(t1 - t0)))

    return test_result

def pack_gbrg_raw_for_compute_ssim(raw):

    im = raw.astype(np.float32)
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[0:H:2, 0:W:2, :]), axis=2)
    return out

def compute_ssim_for_packed_raw(raw1, raw2):
    raw1_pack = pack_gbrg_raw_for_compute_ssim(raw1)
    raw2_pack = pack_gbrg_raw_for_compute_ssim(raw2)
    test_raw_ssim = 0
    for i in range(4):
        test_raw_ssim += compare_ssim(raw1_pack[:,:,i], raw2_pack[:,:,i], data_range=1.0)

    return test_raw_ssim/4
