import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


#Blind2Unblind
def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


# def depth_to_space(x, block_size):
#     """
#     Input: (N, C × ∏(kernel_size), L)
#     Output: (N, C, output_size[0], output_size[1], ...)
#     """
#     n, c, h, w = x.size()
#     x = x.reshape(n, c, h * w)
#     folded_x = torch.nn.functional.fold(
#         input=x, output_size=(h*block_size, w*block_size), kernel_size=block_size, stride=block_size)
#     return folded_x


def depth_to_space(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


def generate_mask(img, width=4, mask_type='random'):
    # This function generates random masks with shape (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask = torch.zeros(size=(n * h // width * w // width * width**2, ),
                       dtype=torch.int64,
                       device=img.device)
    idx_list = torch.arange(
        0, width**2, 1, dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(size=(n * h // width * w // width, ),
                         dtype=torch.int64,
                         device=img.device)

    if mask_type == 'random':
        rd_idx = torch.randint(low=0,
                      high=len(idx_list),
                      size=(n * h // width * w // width, ),
                      device=img.device,
                      generator=get_generator(device=img.device),
                      out=rd_idx)
    elif mask_type == 'batch':
        '''rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(n, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(h // width * w // width)'''
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(n, ),
                               device=img.device,
                               generator=get_generator()).repeat(h // width * w // width)
    elif mask_type == 'all':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(1, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(n * h // width * w // width)
    elif 'fix' in mask_type:
        index = mask_type.split('_')[-1]
        index = torch.from_numpy(np.array(index).astype(
            np.int64)).type(torch.int64)
        rd_idx = index.repeat(n * h // width * w // width).to(img.device)

    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // width * w // width * width**2,
                                step=width**2,
                                dtype=torch.int64,
                                device=img.device)

    mask[rd_pair_idx] = 1

    mask = depth_to_space(mask.type_as(img).view(
        n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width).type(torch.int64)

    return mask


def interpolate_mask(tensor, mask, mask_inv):
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1)

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv


def interpolate_tensor(tensor):
    n, c, h, w = tensor.shape
    device = tensor.device
    #mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1)

    return filtered_tensor.view_as(tensor)

class Masker(object):
    def __init__(self, width=4, mode='interpolate', mask_type='all'):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n,self.width**2,c,h,w), device=img.device)
        masks = torch.zeros((n,self.width**2,1,h,w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:,i,...] = x
            masks[:,i,...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks

class Masker2(object):
    def __init__(self, width=4, mode='interpolate', mask_type='all'):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n,c,h,w), device=img.device)
        masks = torch.zeros((n,1,h,w), device=img.device)
        for i in range(n):
            mode_i = np.random.randint(self.width**2)
            #print(mode_i)
            x, mask = self.mask(img[i:i+1,:,:,:], mask_type='fix_{}'.format(mode_i))
            tensors[i,:,:,:] = x
            masks[i,:,:,:] = mask
        #tensors = tensors.view(-1, c, h, w)
        #masks = masks.view(-1, 1, h, w)
        return tensors, masks

class Masker2_1(object):
    def __init__(self, width=4):
        self.width = width

    def mask(self, img):
        choice = np.random.randint(2)
        if choice == 0:
            circulation=1
        elif choice == 1:
            circulation=2
        n, c, h, w = img.shape
        mask_inv_final = torch.ones(n,1,h,w).to(img.device)
        for j in range(circulation):
            mode_i = np.random.randint(self.width**2)
            mask_type='fix_{}'.format(mode_i)
            # This function generates masked images given random masks
            mask = generate_mask(img, width=self.width, mask_type=mask_type)
            mask_inv = torch.ones(mask.shape).to(img.device) - mask
            mask_inv_final = mask_inv_final*mask_inv
        mask_final = torch.ones(mask.shape).to(img.device) - mask_inv_final
        
        masked = interpolate_mask(img, mask_final, mask_inv_final)

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n,c,h,w), device=img.device)
        masks = torch.zeros((n,1,h,w), device=img.device)
        for i in range(n):
            x, mask = self.mask(img[i:i+1,:,:,:])
            tensors[i,:,:,:] = x
            masks[i,:,:,:] = mask
        #tensors = tensors.view(-1, c, h, w)
        #masks = masks.view(-1, 1, h, w)
        return tensors, masks

class Masker2_2(object):
    def __init__(self, width=4):
        self.width = width

    def mask(self, img):
        choice = np.random.randint(3)
        if choice == 0:
            circulation=1
        elif choice == 1:
            circulation=2
        elif choice == 2:
            circulation=3
        n, c, h, w = img.shape
        mask_inv_final = torch.ones(n,1,h,w).to(img.device)
        for j in range(circulation):
            mode_i = np.random.randint(self.width**2)
            mask_type='fix_{}'.format(mode_i)
            # This function generates masked images given random masks
            mask = generate_mask(img, width=self.width, mask_type=mask_type)
            mask_inv = torch.ones(mask.shape).to(img.device) - mask
            mask_inv_final = mask_inv_final*mask_inv
        mask_final = torch.ones(mask.shape).to(img.device) - mask_inv_final
        
        masked = interpolate_mask(img, mask_final, mask_inv_final)

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n,c,h,w), device=img.device)
        masks = torch.zeros((n,1,h,w), device=img.device)
        for i in range(n):
            x, mask = self.mask(img[i:i+1,:,:,:])
            tensors[i,:,:,:] = x
            masks[i,:,:,:] = mask
        #tensors = tensors.view(-1, c, h, w)
        #masks = masks.view(-1, 1, h, w)
        return tensors, masks

class Masker2_3(object):
    def __init__(self, width=4):
        self.width = width

    def mask(self, img):
        choice = np.random.randint(4)
        if choice == 0:
            circulation=1
        elif choice == 1:
            circulation=2
        elif choice == 2:
            circulation=3
        elif choice == 3:
            circulation=4
        n, c, h, w = img.shape
        mask_inv_final = torch.ones(n,1,h,w).to(img.device)
        for j in range(circulation):
            mode_i = np.random.randint(self.width**2)
            mask_type='fix_{}'.format(mode_i)
            # This function generates masked images given random masks
            mask = generate_mask(img, width=self.width, mask_type=mask_type)
            mask_inv = torch.ones(mask.shape).to(img.device) - mask
            mask_inv_final = mask_inv_final*mask_inv
        mask_final = torch.ones(mask.shape).to(img.device) - mask_inv_final
        
        masked = interpolate_mask(img, mask_final, mask_inv_final)

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n,c,h,w), device=img.device)
        masks = torch.zeros((n,1,h,w), device=img.device)
        for i in range(n):
            x, mask = self.mask(img[i:i+1,:,:,:])
            tensors[i,:,:,:] = x
            masks[i,:,:,:] = mask
        #tensors = tensors.view(-1, c, h, w)
        #masks = masks.view(-1, 1, h, w)
        return tensors, masks

class Masker3(object):
    def __init__(self, mode='interpolate', mask_type='all'):
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, width, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        choice = np.random.randint(2)
        if choice==0:
            width = 4
        else:
            width = 2
        n, c, h, w = img.shape
        tensors = torch.zeros((n,c,h,w), device=img.device)
        masks = torch.zeros((n,1,h,w), device=img.device)
        for i in range(n):
            mode_i = np.random.randint(width**2)
            #print(mode_i)
            x, mask = self.mask(img[i:i+1,:,:,:], width=width, mask_type='fix_{}'.format(mode_i))
            tensors[i,:,:,:] = x
            masks[i,:,:,:] = mask
        #tensors = tensors.view(-1, c, h, w)
        #masks = masks.view(-1, 1, h, w)
        return tensors, masks

class Masker4(object):
    def __init__(self, mode='interpolate', mask_type='all'):
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, width, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        choice = np.random.randint(3)
        if choice==0:
            width = 2
        elif choice==1:
            width = 4
        else:
            width = 8
        n, c, h, w = img.shape
        tensors = torch.zeros((n,c,h,w), device=img.device)
        masks = torch.zeros((n,1,h,w), device=img.device)
        for i in range(n):
            mode_i = np.random.randint(width**2)
            #print(mode_i)
            x, mask = self.mask(img[i:i+1,:,:,:], width=width, mask_type='fix_{}'.format(mode_i))
            tensors[i,:,:,:] = x
            masks[i,:,:,:] = mask
        #tensors = tensors.view(-1, c, h, w)
        #masks = masks.view(-1, 1, h, w)
        return tensors, masks

#Neighbor2Neighbor
operation_seed_counter = 0

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage

#Noise2Void
def generate_n2v_input(input):

    ratio = 0.9
    size_window = (5, 5)
    size_data = (128,128,4)
    num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

    output = input.clone()

    idy_msk = np.random.randint(0, size_data[0], num_sample)
    idx_msk = np.random.randint(0, size_data[1], num_sample)

    idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
    idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

    idy_msk_neigh = idy_msk + idy_neigh
    idx_msk_neigh = idx_msk + idx_neigh

    idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
    idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

    id_msk = (idy_msk, idx_msk)
    id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

    output[:,:,id_msk] = input[:,:,id_msk_neigh]

    return output