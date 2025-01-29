#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, TensorDataset,  random_split
import pandas as pd
import matplotlib.image as mpimg
from torchvision import transforms
import cv2
import sys
import re


import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import logging

from torch import nn, Tensor
from collections import OrderedDict
import copy
from typing import Optional, List

#import yaml
import torchvision.models as models
import matplotlib.patches as mpatches



os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


PARAM_EPOCHS = int(sys.argv[1])
PARAM_BATCH = int(sys.argv[2])
PARAM_H = int(sys.argv[3])
PARAM_W = int(sys.argv[3])
PARAM_LOSS = sys.argv[4]


STAT = {'png': 0, 'jpg': 0, 'grayscale': 0}

def init_gpu():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print("Using MPS Apple GPU.")
        return mps_device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
        return device
    else:
        device = torch.device("cpu")
        print("CUDA/MPS not available. Using CPU.")
        return device



def extract_number(file_path):
    # Estrae il numero dal nome del file utilizzando un'espressione regolare
    match = re.search(r'(\d+)', file_path)
    if match:
        return int(match.group(1))
    return 0


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class JointsHuberLoss(nn.Module): # https://www.geeksforgeeks.org/what-are-some-common-loss-functions-used-in-training-computer-vision-models/  https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss
    def __init__(self, use_target_weight):
        super(JointsHuberLoss, self).__init__()
        self.criterion = nn.HuberLoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class KeypointsDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Costruisco la lista dei campioni validi
        file_list = os.listdir(self.root_dir)
        for filename in file_list:
            if (filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".jpeg") or filename.endswith(".Jpeg") or filename.endswith(".JPEG") or filename.endswith(".PNG") or filename.endswith(".png")) and ("-1" not in filename and " 2" not in filename):
                idx = filename.split('.')[0]
                txt_name = f"{idx}.txt"
                if not os.path.exists(os.path.join(self.root_dir, txt_name)):
                    txt_name = f"{idx} .txt"
                txt_path = os.path.join(self.root_dir, txt_name)
                img_path = os.path.join(self.root_dir, filename)

                if (filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".jpeg") or filename.endswith(".Jpeg") or filename.endswith(".JPEG")):
                    STAT['jpg'] += 1
                else:
                    STAT['png'] += 1

                # Controllo per escludere immagini che hanno più di 14 punti di interesse 
                # o meno di 14 punti all'interno del file delle coordinate .txt
                if os.path.exists(txt_path):
                    data = pd.read_csv(txt_path, delimiter='\t' or ',')
                    if len(data.index) == 14:
                        self.samples.append((img_path, txt_path))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]
        image = mpimg.imread(img_path)

        try:
            data = pd.read_csv(txt_path, delimiter='\t')
            coordinates = data[['X', 'Y']].values
        except KeyError:
            data = pd.read_csv(txt_path, delimiter=',')
            coordinates = data[['X', 'Y']].values

        sample = {'image': image, 'keypoints': coordinates}

        if self.transform:
            sample = self.transform(sample)
        return sample



class Normalize(object):
    """ Normalizzo per migliorare l'apprendimento"""
    def __init__(self, mean, std, H, W):
        self.mean = mean
        self.std = std
        self.H = H
        self.W = W

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        #Converto immagine a colori
        if len(image.shape) > 2:
            image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_copy = image
            STAT['grayscale'] += 1

        # Normalizzo le immagini
        image_copy = (image_copy - self.mean) / self.std

        # Normalizzo le coordinate in base alle dimensioni desiderate
        key_pts_copy[:,0] = key_pts_copy[:,0]/self.W
        key_pts_copy[:,1] = key_pts_copy[:,1]/self.H

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """ Ridimensiona le immagini e i relativi punti"""
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]

        img = cv2.resize(image, (self.W, self.H))
        key_pts = key_pts * [self.W/ w, self.H / h]

        return {'image': img, 'keypoints': key_pts}


class ToTensor(object):
    """ Converto immagini e punti in tensori per Pytorch"""
    def __call__(self, sample):
        if not sample:
            return {}
        image, key_pts = sample['image'], sample['keypoints']


        if(len(image.shape) == 2):
            # aggiungo una terza dimensione per lavorare con le CNN
            image = image.reshape(image.shape[0], image.shape[1], 1)

        #Converto in (C, H, W)
        image = image.transpose((2, 0, 1))

        # Converto i keypoints in un array NumPy se sono una lista
        if isinstance(key_pts, list):
            key_pts = np.array(key_pts)
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}


class RandomRotation(object):
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        rand = np.random.uniform(0, 1)

        if rand <= 0.25:
            flipped_image = np.fliplr(image)
            key_pts[:, 0] = image.shape[1] - key_pts[:, 0]

            return {'image': flipped_image, 'keypoints': key_pts}

        elif rand <= 0.5:
            angle =  90
        elif rand <= 0.75:
            angle = 180
        else:
            angle = 270

        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # Ruotazione immagine
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_image = cv2.warpAffine(image, M, (w, h))

        # Ruotazione keypoints
        ones = np.ones(shape=(len(key_pts), 1))
        points_ones = np.hstack([key_pts, ones])
        transformed_points = M.dot(points_ones.T).T

        return {'image': rotated_image, 'keypoints': transformed_points}


class RandomFilterOrContrast(object):
    def __init__(self, kernel, alpha, beta, probability):
        self.kernel = kernel
        self.probability = probability
        self.alpha = alpha
        self.beta = beta

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        if np.random.rand() < self.probability:

            # Applico il filtro personalizzato per i bordi
            filtered_image = cv2.filter2D(image, -1, kernel=self.kernel)

        elif np.random.rand() < 0.8:
            #Rendo l'immagine più luminosa (beta)
            filtered_image = cv2.convertScaleAbs(image,  self.alpha, self.beta)

        else:
            #Riduco la luminosità dell'immagine
            filtered_image = np.clip(image.astype(np.int16) - self.beta, 0, 255).astype(np.uint8)

        return {'image': filtered_image, 'keypoints': key_pts}



def denormalize_keypoints(keypoints_normalized, width, height):

    # Faccio un controllo per vedere in che formato sono (B, H, W) o (H, W)
    if keypoints_normalized.dim() == 3:
        keypoints_original_x = keypoints_normalized[:, :, 0] * (width - 1)
        keypoints_original_y = keypoints_normalized[:, :, 1] * (height - 1)
    else:
        keypoints_original_x = keypoints_normalized[:, 0] * (width - 1)
        keypoints_original_y = keypoints_normalized[:, 1] * (height - 1)

    keypoints_original = torch.stack([keypoints_original_x, keypoints_original_y], dim=-1)
    return keypoints_original



################################### M O D E L L O ##################################################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, BN_MOMENTUM, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, BN_MOMENTUM, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, bn_momentum, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.bn_momentum = bn_momentum
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels, bn_momentum)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         BN_MOMENTUM, stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=self.bn_momentum
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                BN_MOMENTUM,
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    BN_MOMENTUM
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, BN_MOMENTUM):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels, BN_MOMENTUM)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class TransKeyH(nn.Module):

    blocks_dict = {
        'BASIC': BasicBlock,
        'BOTTLENECK': Bottleneck
    }

    def __init__(self, BN_MOMENTUM, W, H, use_cpu = False):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.pretrained_layers = ['*']

        self.use_cpu = use_cpu

        d_model = 512
        dim_feedforward = 1024
        encoder_layers_num = 6
        n_head = 8
        pos_embedding_type = 'sine'
        w, h = [W, H]

        super(TransKeyH, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)


        self.layer1 = self._make_layer(Bottleneck, 64, 4, BN_MOMENTUM)
        #self.layer1 = self._make_layer(Bottleneck, 64, 3, BN_MOMENTUM)
        #self.layer2 = self._make_layer(Bottleneck, 128, 4, BN_MOMENTUM, stride=2)
        #self.reduce = nn.Conv2d(self.inplanes, d_model, 1, bias=False)


        num_channels = [32, 64]
        block = self.blocks_dict['BASIC']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            {'NUM_MODULES':1, 'NUM_BRANCHES': 2, 'NUM_BLOCKS': [4,4],
             'NUM_CHANNELS': [32,64], 'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'},
            num_channels)

        num_channels = [32, 64, 128]
        block = self.blocks_dict['BASIC']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            {'NUM_MODULES':1, 'NUM_BRANCHES': 3, 'NUM_BLOCKS': [4,4,4],
             'NUM_CHANNELS': [32,64, 128], 'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'},
            num_channels, multi_scale_output=False)

        #num_channels = [32, 64, 128, 256]
        #block = self.blocks_dict['BASIC']
        #num_channels = [
        #    num_channels[i] * block.expansion for i in range(len(num_channels))
        #]
        #self.transition3 = self._make_transition_layer(
        #    pre_stage_channels, num_channels)
        #self.stage4, pre_stage_channels = self._make_stage(
        #    {'NUM_MODULES':1, 'NUM_BRANCHES': 4, 'NUM_BLOCKS': [4,4,4,4],
        #     'NUM_CHANNELS': [32, 64, 128, 256], 'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'},
        #    num_channels, multi_scale_output=False)



        self.reduce = nn.Conv2d(pre_stage_channels[0], d_model, 1, bias=False)
        self._make_position_embedding(w, h, d_model, pos_embedding_type)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            activation='relu'
        )

        self.global_encoder = TransformerEncoder(
            encoder_layer,
            encoder_layers_num
        )

        self.inplanes = d_model

        self.deconv_layers = self._make_deconv_layer(
            1,          # Layers
            [d_model],  # Filtri
            [2],        # Kernel
            BN_MOMENTUM
        )

        self.final_layer = nn.Conv2d(
            in_channels=d_model,
            out_channels= 14,
            kernel_size= 1,
            stride=1,
            padding=0
        )


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 4
                self.pe_w = w // 4
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000):
        h, w = self.pe_h, self.pe_w
        grid_h, grid_w = torch.meshgrid(
            torch.linspace(0, h-1, h), torch.linspace(0, w-1, w), indexing='ij'
        )
        grid_h = grid_h.reshape(-1, 1)
        grid_w = grid_w.reshape(-1, 1)

        # Calcola le posizioni sinusoidali
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(np.log(temperature) / d_model))
        pos_w = torch.zeros((w * h, d_model), dtype=torch.float32)
        pos_w[:, 0::2] = torch.sin(grid_w * div_term)
        pos_w[:, 1::2] = torch.cos(grid_w * div_term)

        pos_h = torch.zeros((w * h, d_model), dtype=torch.float32)
        pos_h[:, 0::2] = torch.sin(grid_h * div_term)
        pos_h[:, 1::2] = torch.cos(grid_h * div_term)

        # Combina le posizioni H e W
        pos_embedding = pos_w + pos_h

        return pos_embedding

    def _make_sine_position_embedding_original(self, d_model, temperature=10000,
                                               scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, BN_MOMENTUM, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, BN_MOMENTUM, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, BN_MOMENTUM))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding



    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, BN_MOMENTUM):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = self.blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def find_keypoints(self, heatmaps):
        batch_size, num_keypoints, height, width = heatmaps.size()

        # Faccio reshape le mappe di calore per semplificare l'operazione di argmax
        heatmaps_reshaped = heatmaps.view(batch_size, num_keypoints, -1)

        # Trova le coordinate (indice) del valore massimo (quindi più caldo) lungo l'asse dell'ultimo canale
        keypoints_flat = torch.argmax(heatmaps_reshaped, dim=-1)

        # Numero di punti di attivazione alti da trovare
        K = 8
        if self.use_cpu:
            heatmaps_cpu = heatmaps.detach().cpu().numpy()
        else:
            heatmaps_cpu = heatmaps.detach().numpy()

        # Trovo gli indici dei K valori più alti (usati per verifica stampa)
        indices = np.unravel_index(np.argsort(heatmaps_cpu.ravel())[-K:], heatmaps_cpu.shape)
        valori_attivazione = heatmaps_cpu[indices]

        # Calcolo le coordinate 2D (y, x) a partire dagli indici piatti
        keypoints_y = keypoints_flat // width
        keypoints_x = keypoints_flat % width

        # Normalizzo le coordinate in [0, 1]
        keypoints_x = keypoints_x.float() / (width - 1)
        keypoints_y = keypoints_y.float() / (width - 1)

        # Restituisco le coordinate dei punti chiave
        keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)


        return keypoints, valori_attivazione

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.reduce(x)

        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x = self.reduce(y_list[0])
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)

        # Applicazione del positional encoding
        if self.pos_embedding is not None:
            pos_embed = self.pos_embedding.view(h * w, 1, c).repeat(1, bs, 1)
            x = x + pos_embed

        # Passaggio dall'encoder del Trasformer
        x = self.global_encoder(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Reshape e passaggio al layer finale
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        x = self.deconv_layers(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer finale per produrre le mappe di calore
        hm = self.final_layer(x)
        hm = torch.sigmoid(hm)

        # Estrazione dei punti chiave dalle mappe di calore
        x, max_v = self.find_keypoints(hm)

        return x, hm, max_v

    def init_weights(self, pretrained='', print_load_info=False):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            if self.use_cpu:
                pretrained_state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
            else:
                pretrained_state_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))

            existing_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers and name in self.state_dict() \
                        or self.pretrained_layers[0] == '*':
                    if name.split('.')[0] == 'conv1':
                        mean_conv1_weights = m.mean(dim=1, keepdim=True)
                        existing_state_dict[name] = mean_conv1_weights
                    else:
                        existing_state_dict[name] = m
                    if print_load_info:
                        print(":: {} is loaded from {}".format(name, pretrained))
            self.load_state_dict(existing_state_dict, strict=False)
        elif pretrained:
            print('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


#################################### F I N E - M O D E L L O ###############################################



def gaussian(x, y, H, W, sigma=5, device='cpu'):
    xs = torch.arange(0, W, step=1, dtype=torch.float32).to(device)
    ys = torch.arange(0, H, step=1, dtype=torch.float32).to(device)
    ys = ys[:, None]

    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    channel = torch.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return channel.float()


def generate_heatmaps_batch(batch, H, W):
    """
    Genera mappe di calore per un batch di keypoints.
    """
    images = batch['image']
    keypoints = batch['keypoints'].float()
    keypoints = denormalize_keypoints(keypoints, W, H)
    batch_size = keypoints.shape[0]
    num_keypoints = keypoints.shape[1]
    heatmaps_batch = torch.zeros((batch_size, num_keypoints, H, W), dtype=torch.float32)

    for b in range(batch_size):
        for i in range(0, num_keypoints):
            x = keypoints[b, i , 0].item()
            y = keypoints[b, i, 1].item()
            heatmaps_batch[b, i] = gaussian(x, y, H, W)

    return images, heatmaps_batch


def stampa_mappa(heatMap, image, t, use_cpu = False):

    if use_cpu:
        hm = heatMap.cpu().detach().numpy()
        image = image.cpu().detach().numpy()[0]
    else:
        hm = heatMap.detach().numpy()
        image = image.detach().numpy()[0]

    nrows = 4  # Numero di righe di sottotrame
    ncols = 4  # Numero di colonne di sottotrame

    img_index = 0  # Indice dell'immagine che visualizzo da esempio


    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
    target_size = image[img_index].shape[:2]  # Dimensione dell'immagine
    #print("target_size:   ", target_size)


    for idx, ax in enumerate(axs.flat):
            # Controllo per evitare indici fuori range
                if idx < hm.shape[1]:

                    heatmap = hm[img_index, idx]
                    heatmap_resized = cv2.resize(heatmap, (target_size), interpolation=cv2.INTER_LINEAR)

                    ax.imshow(image[img_index], cmap='gray')

                    ax.imshow(heatmap_resized, cmap='viridis', interpolation='bilinear', alpha=0.7)
                    ax.set_title(f'Keypoint {idx+1}')
                    ax.axis('off')

    plt.tight_layout()

    # t parametro per sapere cosa salvare con nomi diversi
    if t == 0:
        plt.savefig('hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/HeatMapValPred-hrnet-epoch{}-batch{}-size{}-loss{}.png'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))
    elif t == 1:
        plt.savefig('hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/HeatMapValTrue-hrnet-epoch{}-batch{}-size{}-loss{}.png'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))
    elif t == 2:
        plt.savefig('hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/HeatMapTrainTrue-hrnet-epoch{}-batch{}-size{}-loss{}.png'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))
    elif t == 3:
        plt.savefig('hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/HeatMapTrainPred-hrnet-epoch{}-batch{}-size{}-loss{}.png'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))

    plt.close()


def draw_gaussian(heatmap, center, radius):
    diameter = 2 * radius + 1
    gaussian = cv2.getGaussianKernel(diameter, -1)
    gaussian = np.outer(gaussian, gaussian)

    gaussian /= gaussian.max()

    x, y = center
    x, y = int(x.item()), int(y.item())

    height, width = heatmap.shape[0:2]

    left = max(x - radius, 0)
    right = min(x + radius + 1, width)
    top = max(y - radius, 0)
    bottom = min(y + radius + 1, height)

    # Calcola le dimensioni effettive della regione su cui disegnare la gaussiana
    width_gaussian = right - left
    height_gaussian = bottom - top

    # Verifica che la regione non sia di dimensioni nulle
    if width_gaussian > 0 and height_gaussian > 0:
        masked_heatmap = heatmap[top:bottom, left:right]
        masked_gaussian = gaussian[radius - (y - top):radius + (bottom - y), radius - (x - left):radius + (right - x)]
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def create_heatmaps_batch(true_keypoints_batch,height, width, num_keypoints, radius):

    batch_size = true_keypoints_batch.shape[0]
    heatmaps_batch = np.zeros((batch_size, num_keypoints,height, width ), dtype=np.float32)

    for img_index in range(batch_size):
        # Estraggo i keypoints per l'immagine corrente
        true_keypoints = true_keypoints_batch[img_index]

        # Crea le mappe di calore per l'immagine corrente
        heatmaps = create_heatmaps(true_keypoints,height, width, num_keypoints, radius)

        # Aggiungi le mappe di calore al batch
        heatmaps_batch[img_index] = heatmaps

    return heatmaps_batch


def create_heatmaps(true_keypoints,height, width, num_keypoints, radius):
    heatmaps = np.zeros((num_keypoints,height, width), dtype=np.float32)

    # Denormalizza i keypoints
    true_keypoints = denormalize_keypoints(true_keypoints, width, height)

    for i in range(num_keypoints):
        # Estraggo le coordinate x e y del keypoint corrente
        x = true_keypoints[i, 0]
        y = true_keypoints[i, 1]

        # Disegno la mappa di calore centrata sul singolo punto di un raggio pari a "radius"
        draw_gaussian(heatmaps[i], (x, y), radius)

    return heatmaps



def calcola_mre(keypoints_predetti, keypoints_verita):
    # Calcola la distanza euclidea tra i keypoints predetti e quelli veri
    distanze = torch.norm(keypoints_predetti - keypoints_verita, dim=2)

    # Calcola l'MRE come la media delle distanze
    mre = distanze.mean().item()

    return mre


def calcola_sdr(keypoints_predetti, keypoints_verita, soglia):
    # Calcola la distanza euclidea tra i keypoints predetti e quelli veri
    distanze = torch.norm(keypoints_predetti - keypoints_verita, dim=2)

    # Conta i keypoints per cui la distanza è inferiore alla soglia
    keypoints_corretti = (distanze < soglia).float()

    # Calcola l'SDR come la percentuale di keypoints corretti
    sdr = keypoints_corretti.mean().item() * 100  # Converti in percentuale

    return sdr

def compute_range(num_epoc, init, fin=1):
    """ 
    Funzione per descrementare il raggio della mappa di calore
    dopo tot epoche
    """
    if num_epoc % 10 == 0:
        new_init = max(init - 2, fin)
    else:
        new_init = init
    return new_init


def plot_keypoints_pred(ax,image, true_keypoints, predicted_keypoints, H, W):

    if image.dim() == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    # Sposta il tensore sulla CPU 
    ax.imshow(image.cpu().numpy(), cmap='gray')

    # Disegna i keypoints veri
    true_keypoints = denormalize_keypoints(true_keypoints, W, H)
    for k in range(true_keypoints.shape[0]):
        ax.scatter(true_keypoints[k, 0].cpu().numpy(), true_keypoints[k, 1].cpu().numpy(), c='red', s=10, label='True' if k == 0 else "")

    # Disegna i keypoints predetti
    predicted_keypoints = denormalize_keypoints(predicted_keypoints, W, H)
    for k in range(predicted_keypoints.shape[0]):
        ax.scatter(predicted_keypoints[k, 0].cpu().numpy(), predicted_keypoints[k, 1].cpu().numpy(), c='blue', s=10, label='Predicted' if k == 0 else "")


def stampa_metric_key(metric_key):
    # Converto la lista di array in un unico array 
    metric_key_array = np.stack(metric_key)

    # Ora metric_key_array ha forma [epoche, keypoints]
    num_keypoints = metric_key_array.shape[1]
    num_epochs = metric_key_array.shape[0]

    plt.figure(figsize=(15, 10))
    for i in range(num_keypoints):
        plt.scatter(range(1, num_epochs + 1), metric_key_array[:, i], label=f'KeyPoint {i+1}')
        m, b = np.polyfit(range(1, num_epochs + 1), metric_key_array[:, i], 1)
        plt.plot(range(1, num_epochs + 1), m * np.arange(1, num_epochs + 1) + b)

    plt.xlabel('Epoca')
    plt.ylabel('Metrica')
    plt.title('Metriche per KeyPoint attraverso le Epoca')
    plt.legend()
    plt.savefig('hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/graficoKey-hrnet-epoch{}-batch{}-size{}-loss{}.png'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))
    plt.close()









# ==================================================

def get_there_model(model_path, BN_MOMENTUM, W, H, use_cpu = False):
    if os.path.isfile(model_path):

        # Carico il modello TransKeyH
        model = TransKeyH(BN_MOMENTUM, W, H, use_cpu)

        # Carico i pesi del modello
        state_dict = torch.load(model_path) # TODO GPU?
        model.load_state_dict(state_dict)
        model.eval()
        print("Loaded model from {}".format(model_path))

    else:

        # Carico il modello TransKeyH
        model = TransKeyH(BN_MOMENTUM, W, H, use_cpu)

        # Salva i pesi originali del modello
        original_state_dict = copy.deepcopy(model.state_dict())
        #pretrained_model= model.init_weights('ResNet50-Download.pth')

        # carico da file i pesi pre addestrati su IMAGENET1K_V1
        # (Su cluster Coka non posso utilizzare il comando "weights=ResNet50_Weights.IMAGENET1K_V1")
        path_to_pth_file = 'hrnet_w32.pth'
        #pretrained_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.init_weights(path_to_pth_file,True)



        loaded_state_dict = model.state_dict()

        # Confronto i pesi prima e dopo il caricamento per capire che i pesi sono stati effettivamente caricati
        for key in original_state_dict:
            if key in loaded_state_dict:
                if not torch.equal(original_state_dict[key], loaded_state_dict[key]):
                    print(f"Weights changed for layer: {key}")
            else:
                print(f"Layer {key} was not found in the loaded model.")
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Trainable: {param.requires_grad}")

    return model









def main():
    mps_device = init_gpu()

    use_cpu = False
    if mps_device.type == 'cpu':
        use_cpu = True

    H, W = PARAM_H, PARAM_W
    BATCH = PARAM_BATCH
    EPOCHS = PARAM_EPOCHS
    BN_MOMENTUM = 0.1

    #Posizione immagini
    img_root = 'dataset_merged'

    #valore medio e dev standard di ImageNet
    mean = 0.485 * 255
    std = 0.229 * 255

    data_transform =transforms.Compose(
    [
        Rescale(H, W),
        Normalize(mean, std, H, W),
        ToTensor()
    ])

    #Creazione Dataset 
    transformed_dataset = KeypointsDataset(root_dir=img_root, transform=data_transform)
    print(STAT)



    dataset_size = len(transformed_dataset)
    print("Dataset tot: ", dataset_size)
    train_size = int(0.7 * dataset_size)
    print("Train size: ", train_size)
    val_size = int(0.15 * dataset_size)
    print("Val size: ", val_size)
    test_size = dataset_size - train_size - val_size
    print("Test size: ", test_size)

    train_dataset, val_dataset, test_dataset = random_split(transformed_dataset, [train_size, val_size, test_size])


    # Creo un DataLoader per esplorare i dati
    #train_loader = DataLoader(transformed_dataset, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


    # Maschera utilizzata per accentuare i bordi durante DataAugmentation
    custom_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

    transform3 =transforms.Compose(
    [
        Rescale(H, W),
        RandomRotation(),
        RandomFilterOrContrast(custom_kernel, 1, 1.3, probability=0.5),
        Normalize(mean, std, H, W),
        ToTensor()
    ])


    dataset_dataAug = KeypointsDataset(root_dir=img_root, transform=transform3)


    print("Dataset tot: ", dataset_size)
    train_size = int(0.6 * dataset_size)
    print("Train size Augmentato: ", train_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_datasetA, _, _ = random_split(dataset_dataAug, [train_size, val_size, test_size])

    # DataLoader contenente le immagini di train modificate
    # che verranno aggiunte a quelle normali
    train_loaderA = DataLoader(train_datasetA, batch_size=BATCH, shuffle=True)
    print("Train_loader Augmentato: ", len(train_loaderA))


    # Unisco dataset di train con immagini originali e dataset con immagini modificate
    combined_dataset = ConcatDataset([train_datasetA, train_dataset])
    print("Dataset di train tot: ", len(combined_dataset))
    train_dataAug = DataLoader(combined_dataset, batch_size=BATCH, shuffle=True)


    model = get_there_model('hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/best_saved_model_TransKeyH-epoch{}-batch{}-size{}-loss{}.pth'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS), BN_MOMENTUM, W, H, use_cpu)


    # Assegno il modello al device
    model.to(mps_device)
    print("Model params:{:.3f}M".format(sum([p.numel() for p in model.parameters()])/1000**2))


    if PARAM_LOSS == 'l1':
        criterion = nn.L1Loss()
    elif PARAM_LOSS == 'jointmse':
        criterion = JointsMSELoss(
            use_target_weight=False
        )
    elif PARAM_LOSS == 'huber':
        criterion = nn.HuberLoss()
    elif PARAM_LOSS == 'jointhuber':
        criterion = JointsHuberLoss(
            use_target_weight=False
        )
    elif PARAM_LOSS == 'mse':
        criterion = nn.MSELoss()
    else:
        print('loss not set')
        exit()
    #criterion = nn.BCELoss()



    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)



    num_epochs = EPOCHS #Numero di epoche
    train_losses = []
    val_losses = []
    metric_history = []
    metric_mre = []
    metric_sdr_at05 = []
    metric_sdr_at1 = []
    metric_sdr_at5 = []
    metric_key = []



    # Parametri per l'early stopping
    patience = 8
    best_val_loss = float("inf")
    epochs_no_improve = 0  # Contatore delle epoche senza miglioramento
    early_stop = False  # Flag per interrompere l'addestramento
    best_model_state = None


    model = model.to(mps_device)

    for epoch in range(num_epochs):
            distanze_totali = []

            # Modello in modalità di addestramento
            model.train()

            total_train_loss = 0

            for batch_index, batch in enumerate(train_dataAug):
                if not batch:
                    continue

                inputs = batch['image'].float().to(mps_device)
                true_keypoints = batch['keypoints'].float().to(mps_device)

                true_heatmaps =  create_heatmaps_batch(true_keypoints, int(PARAM_H/2), int(PARAM_W/2), 14, 6)
                true_heatmaps = torch.from_numpy(true_heatmaps).float().to(mps_device)

                # Azzero i gradienti dell'ottimizzatore
                optimizer.zero_grad()

                outputs, hm, _ = model(inputs)
                hm = hm.float().to(mps_device)
                #print(hm.shape)

                loss = (criterion(hm, true_heatmaps))*100
                total_train_loss += loss.item()

                sys.stdout.write(f'\rEpoch {epoch+1}/{num_epochs} - Batch {batch_index + 1}/{len(train_dataAug)} - Loss: {loss.item()}')
                sys.stdout.flush()


                #Backward pass e ottimizzazione
                loss.backward()
                optimizer.step()


            train_loss_avg = total_train_loss / len(train_dataAug)
            train_losses.append(train_loss_avg)


            with torch.no_grad():

                # Disabilita il calcolo dei gradienti
                model.eval()

                val_loss = 0
                val_pck = 0
                val_mre = 0
                val_sdr_at05 = 0
                val_sdr_at1 = 0
                val_sdr_at5 = 0

                for batch in val_loader:  # Itera sul DataLoader di validazione

                    inputs = batch['image'].float().to(mps_device)
                    true_keypointsV = batch['keypoints'].float().to(mps_device)

                    true_heatmapsV = create_heatmaps_batch(true_keypointsV, int(PARAM_H/2), int(PARAM_W/2), 14, 6)
                    true_heatmapsV = torch.from_numpy(true_heatmapsV).float().to(mps_device)


                    outputs, hm, _ = model(inputs)

                    outputs = (outputs).float().to(mps_device)

                    loss = (criterion(outputs, true_keypointsV)*100)

                    val_loss += loss.item()

                    true_keypointsV = denormalize_keypoints(true_keypointsV, W, H)
                    outputs = denormalize_keypoints(outputs, W, H)

                    distanze_batch = torch.norm(outputs - true_keypointsV, dim=-1)

                    # Soglia utilizzata per il calcolo della SDR -- 2 pixel
                    soglia_at05 = int(round(PARAM_H * 0.005))
                    soglia_at1 = int(round(PARAM_H * 0.01))
                    soglia_at5 = int(round(PARAM_H * 0.05))
                    punteggi_batch = (distanze_batch <= soglia_at1).float()

                    # Array di distanza per ogni batch
                    distanze_totali.append(punteggi_batch)


                    val_mre += calcola_mre(outputs,true_keypointsV)

                    val_sdr_at05 += calcola_sdr(outputs,true_keypointsV,soglia_at05)
                    val_sdr_at1 += calcola_sdr(outputs,true_keypointsV,soglia_at1)
                    val_sdr_at5 += calcola_sdr(outputs,true_keypointsV,soglia_at5)


            # Concatena le distanze di tutti i batch
            distanze_totali = torch.cat(distanze_totali, dim=0)

            # Calcola la metrica media per ciascun keypoint su tutti i dati
            if use_cpu:
                metriche_per_keypoint = distanze_totali.mean(dim=0).cpu().numpy() *100
            else:
                metriche_per_keypoint = distanze_totali.mean(dim=0).numpy() *100

            # Stampa la metrica media per ciascun keypoint
            print(f"\nPrecisione per ogni keypoint con soglia pari a {soglia_at1} pixel (@1.0%): ")
            for i, metrica in enumerate(metriche_per_keypoint, 1):
                print(f"KeyPoint {i}: {metrica:.2f} %")

            metric_key.append(metriche_per_keypoint)

            # Calcola la loss media di validazione
            val_loss_avg = val_loss / len(val_loader)
            print(f"\n***** Loss media sulla validazione: {val_loss_avg:.5f} *****")

            #val_pck_avg = val_pck / len(val_loader)
            #print(f'***** Metrica PCK: {val_pck_avg:.5f} *****\n\n')
            val_med_mre = val_mre / len(val_loader)
            print(f'***** Metrica MRE: {val_med_mre:.5f} *****\n\n')

            val_med_sdr_at05 = val_sdr_at05 / len(val_loader)
            val_med_sdr_at1 = val_sdr_at1 / len(val_loader)
            val_med_sdr_at5 = val_sdr_at5 / len(val_loader)
            print(f'***** Metrica SDR@0.5%: {val_med_sdr_at05:.5f} *****')
            print(f'***** Metrica SDR@1.0%: {val_med_sdr_at1:.5f} *****')
            print(f'***** Metrica SDR@5.0%: {val_med_sdr_at5:.5f} *****\n\n')

            val_losses.append(val_loss_avg)
            metric_mre.append(val_med_mre)
            metric_sdr_at05.append(val_med_sdr_at05)
            metric_sdr_at1.append(val_med_sdr_at1)
            metric_sdr_at5.append(val_med_sdr_at5)

            # Controllo per early stopping e per tenere i pesi del modello più basso di loss
            if val_loss_avg <= best_val_loss:
                best_val_loss = val_loss_avg
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), 'hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/best_saved_model_TransKeyH-epoch{}-batch{}-size{}-loss{}.pth'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))
            else:
                epochs_no_improve +=1
                if epochs_no_improve == patience:
                    early_stop = True
                    break

    scheduler.step(val_loss_avg)

    stampa_mappa(hm,inputs, 0, use_cpu)
    stampa_mappa(true_heatmapsV,inputs, 1, use_cpu)

    stampa_metric_key(metric_key)
    # Tracciamento delle loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss During Training and Validation')
    plt.legend()
    plt.savefig('hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/graficoTrain-hrnet-epoch{}-batch{}-size{}-loss{}.png'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))
    plt.close()

    # Tracciamento mre
    plt.figure(figsize=(10, 5))
    plt.plot(metric_mre, label = 'mre')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.title('Metric Mean Relative Error')
    plt.legend()
    plt.savefig('hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/graficoMse-hrnet-epoch{}-batch{}-size{}-loss{}.png'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))
    plt.close()

    # Tracciamento sdr
    plt.figure(figsize=(10, 5))
    plt.plot(metric_sdr_at05, label = 'sdr@0.5%')
    plt.plot(metric_sdr_at1, label = 'sdr@1.0%')
    plt.plot(metric_sdr_at5, label = 'sdr@5.0%')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.title('Metric Successful Detection Rate')
    plt.legend()
    plt.savefig('hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/graficoSdr-hrnet-epoch{}-batch{}-size{}-loss{}.png'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))
    plt.close()

    if early_stop:
        print("\nAddestramento interrotto a causa dell'early stopping")
    else:
        print("\nAddestramento completato")

    if best_model_state:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), 'hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/best_saved_model_TransKeyH-epoch{}-batch{}-size{}-loss{}.pth'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))



    counter = 0
    plt.figure(figsize=(15, 15))
    num_images = 36

    model.load_state_dict((best_model_state))
    model.eval()

    for batch in test_loader:
            if counter >= num_images:
                break

            images, _ = generate_heatmaps_batch(batch, H, W)

            img_index = 0  # Indice dell'immagine nel batch

            images = images.float().to(mps_device)
            outputs, _, _= model(images)

            for img_index in range(len(images)):
                ax = plt.subplot(6, 6, counter + 1)  # Creo un subplot per ogni immagine
                plot_keypoints_pred(ax, images[img_index], batch['keypoints'][img_index], outputs[img_index], H, W)

                counter += 1
                if counter >= num_images:
                    break

    red_patch = mpatches.Patch(color='red', label='True')
    blue_patch = mpatches.Patch(color='blue', label='Predicted')

    # Legenda
    plt.legend(handles=[red_patch, blue_patch], loc='upper center', bbox_to_anchor=(-2.5, -0.15), fancybox=True, shadow=True, ncol=2)

    plt.savefig('hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS) + '/TestPrediction-hrnet-epoch{}-batch{}-size{}-loss{}.png'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))
    plt.close()


if __name__ == "__main__":

    newpath = 'hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    print('Test-hrnet-epoch{}-batch{}-size{}-loss{}'.format(PARAM_EPOCHS,PARAM_BATCH,PARAM_H,PARAM_LOSS))
    main()


