#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
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

import yaml
import torchvision.models as models
import matplotlib.patches as mpatches



os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


class KeypointsDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Costruisco la lista dei campioni validi
        file_list = os.listdir(self.root_dir)
        for filename in file_list:
            if (filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".jpeg")) and ("-1" not in filename and " 2" not in filename):
                idx = filename.split('.')[0]
                txt_name = f"{idx}.txt"
                if not os.path.exists(os.path.join(self.root_dir, txt_name)):
                    txt_name = f"{idx} .txt"
                txt_path = os.path.join(self.root_dir, txt_name)
                img_path = os.path.join(self.root_dir, filename)
                

                if os.path.exists(txt_path):
                    data = pd.read_csv(txt_path, delimiter='\t' or ',')
                    if len(data.index) == 14:
                        self.samples.append((img_path, txt_path))
        
        #self.samples = sorted(self.samples, key=lambda x: int(re.findall( r'\d+', os.path.basename(x[0])) [0]) ) 
        #print(self.samples)
                        
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
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
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

class TransKeyR(nn.Module):
    def __init__(self, block, layers, BN_MOMENTUM, W, H):
        super(TransKeyR, self).__init__()
        self.inplanes = 64
        self.deconv_with_bias = False

        d_model = 256 
        dim_feedforward = 1024
        encoder_layers_num = 3
        n_head = 8
        pos_embedding_type = 'learnable'
        w, h = [W, H] 

        super(TransKeyR, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], BN_MOMENTUM)
        self.layer2 = self._make_layer(block, 128, layers[1], BN_MOMENTUM, stride=2)
        
        self.reduce = nn.Conv2d(self.inplanes, d_model, 1, bias=False)
        self._make_position_embedding(w, h, d_model, pos_embedding_type)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            activation='relu',
            batch_first=True
        )

        self.global_encoder = TransformerEncoder(
            encoder_layer,
            encoder_layers_num,
        )

        self.inplanes = d_model
        self.deconv_layers = self._make_deconv_layer(
            1,   #NUM_DECONV_LAYERS 
            [256],  # NUM_DECONV_FILTERS
            [4],  # NUM_DECONV_KERNELS'
            BN_MOMENTUM
        )

        self.final_layer = nn.Conv2d(
            in_channels=d_model,
            out_channels= 14, 
            kernel_size= 1, 
            stride=1,
            padding=0
        )


    def _make_position_embedding(self, w, h, d_model, pe_type='learnable'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
        else:
            with torch.no_grad():
                self.pe_h = h // 8
                self.pe_w = w // 8
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.randn(length, d_model))
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding2(d_model),
                    requires_grad=False)
                



    def _make_sine_position_embedding2(self, d_model, temperature=10000):
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
      


    def _make_layer(self, block, planes, blocks, BN_MOMENTUM, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
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
                    stride=4,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
    


    def find_keypoints(self, heatmaps):
            batch_size, num_keypoints, height, width = heatmaps.size()

            # Faccio reshape le mappe di calore per semplificare l'operazione di argmax
            heatmaps_reshaped = heatmaps.view(batch_size, num_keypoints, -1)

            # Trova le coordinate (indice) del valore massimo (quindi più caldo) lungo l'asse dell'ultimo canale
            keypoints_flat = torch.argmax(heatmaps_reshaped, dim=-1)
            
            # Numero di punti di attivazione alti da trovare
            K = 8  
            heatmaps_cpu = heatmaps.detach().cpu().numpy()

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
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = F.dropout(x, p=0.5, training=self.training)  
        x = self.layer2(x)
        x = F.dropout(x, p=0.5, training=self.training)  
        x = self.reduce(x)

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.global_encoder(x)
        x = F.dropout(x, p=0.5, training=self.training)  
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        x = self.deconv_layers(x)
        x = F.dropout(x, p=0.5, training=self.training)

        hm = self.final_layer(x)
        hm = torch.sigmoid(hm)
        x, max_v = self.find_keypoints(hm)

        return x, hm, max_v


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


def stampa_mappa(heatMap, image, t):

        hm = heatMap.cpu().detach().numpy()
        image = image.cpu().detach().numpy().squeeze()
                
        nrows = 4  # Numero di righe di sottotrame
        ncols = 4  # Numero di colonne di sottotrame
                
        img_index = 0  # Indice dell'immagine che visualizzo da esempio

                
        fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
        target_size = image[img_index].shape[:2]  # Dimensione dell'immagine
        print("target_size:   ", target_size)


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
            plt.savefig('HeatMapValPred.png')
        elif t == 1:
            plt.savefig('HeatMapValTrue.png')
        elif t == 2:
            plt.savefig('HeatMapTrainTrue.png')
        elif t == 3:
            plt.savefig('HeatMapTrainPred.png')
        
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
    plt.savefig('graficoKey.png')
    plt.close()




















def main():
    mps_device = init_gpu()

    H, W = 256, 256
    BATCH=2

    #Posizione immagini
    img_root = 'dataset_teleradiografie_14punti'

    mean = 0.485 * 255
    std = 0.229 * 255

    data_transform =transforms.Compose(
    [
        Rescale(H, W), 
        Normalize(mean, std, H, W),
        ToTensor()
    ])

    #Creazione Dataset 
    transformed_dataset = KeypointsDataset(root_dir=img_root,
                                                transform=data_transform)

    
    dataset_size = len(transformed_dataset)
    print("Dataset tot: ", dataset_size)
    train_size = int(0.7 * dataset_size)
    print("Train size: ", train_size)
    val_size = int(0.15 * dataset_size)
    print("Val size: ", val_size)
    test_size = dataset_size - train_size - val_size
    print("Test size: ", test_size)

    train_dataset, val_dataset, test_dataset = random_split(transformed_dataset, [train_size, val_size, test_size])
    print((train_dataset))


    # Creo un DataLoader per esplorare i dati
    train_loader = DataLoader(transformed_dataset, batch_size=BATCH, shuffle=True)
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


    print("Dataset tot Augmentato: ", dataset_size)
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




    BN_MOMENTUM = 0.1
    
    resnet_spec = { 50: (Bottleneck, [3, 4, 6, 3]),
                    101: (Bottleneck, [3, 4, 23, 3]),
                    152: (Bottleneck, [3, 8, 36, 3])}

    block_class, layers = resnet_spec[50]


    # Carico il modello TransKeyR
    model = TransKeyR(block_class, layers, BN_MOMENTUM, W, H)

    # Salva i pesi originali del modello
    original_state_dict = copy.deepcopy(model.state_dict())
    #pretrained_model= model.init_weights('ResNet50-Download.pth')

    # Crea un'istanza del modello ResNet50 senza pesi preaddestrati
    pretrained_model = models.resnet50()

    # carico da file i pesi pre addestrati su IMAGENET1K_V1
    # (Su cluster Coka non posso utilizzare il comando "weights=ResNet50_Weights.IMAGENET1K_V1")
    path_to_pth_file = 'resnet50.pth'
    #pretrained_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    pretrained_model.load_state_dict(torch.load(path_to_pth_file))

    # Modifico il primo strato convoluzionale per accettare 1 canale invece di 3
    # e calcola la media dei pesi attraverso i canali RGB
    conv1_weights = pretrained_model.conv1.weight.data
    mean_conv1_weights = conv1_weights.mean(dim=1, keepdim=True)

    # Applica la media calcolata al primo strato convoluzionale del modello pre-addestrato
    pretrained_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    pretrained_model.conv1.weight.data = mean_conv1_weights
    
    # Sostituisco i pesi nel modello caricato
    model.load_state_dict(pretrained_model.state_dict(), strict=False)

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


    # Assegno il modello al device
    model.to(mps_device)  
    print("Model params:{:.3f}M".format(sum([p.numel() for p in model.parameters()])/1000**2))

    
    
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    #criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    


    num_epochs = 80 #Numero di epoche
    train_losses = []
    val_losses = []
    metric_history = []
    metric_mre = []
    metric_sdr = []
    metric_key = []
    

    # Parametri per l'early stopping
    patience = 10  
    best_val_loss = float("inf")  
    epochs_no_improve = 0  # Contatore delle epoche senza miglioramento
    early_stop = False  # Flag per interrompere l'addestramento
    best_model_state = None

    # Range per inizializzare la mappa di calore
    current_range = int(12)
    
    model = model.to(mps_device)

    for epoch in range(num_epochs):
            distanze_totali = []
            
            # Modello in modalità di addestramento
            model.train()  
            
            total_train_loss = 0
            #print(current_range)
            #current_range = compute_range(epoch, current_range)
            for batch_index, batch in enumerate(train_dataAug):  
                if not batch:
                    continue
                
                inputs = batch['image'].float().to(mps_device)
                true_keypoints = batch['keypoints'].float().to(mps_device)
                
                true_heatmaps =  create_heatmaps_batch(true_keypoints, 128, 128, 14, 6)
                true_heatmaps = torch.from_numpy(true_heatmaps).float().to(mps_device)
                
                # Azzero i gradienti dell'ottimizzatore
                optimizer.zero_grad()
                
                outputs, hm, indi = model(inputs)
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
                val_sdr = 0

                for batch in val_loader:  # Itera sul DataLoader di validazione

                    inputs = batch['image'].float().to(mps_device)
                    true_keypointsV = batch['keypoints'].float().to(mps_device)

                    true_heatmapsV = create_heatmaps_batch(true_keypointsV,128, 128, 14, 6)
                    true_heatmapsV = torch.from_numpy(true_heatmapsV).float().to(mps_device)

                   
                    outputs, hm, _ = model(inputs)
                    outputs = (outputs).float().to(mps_device)
                    
                    loss = (criterion(outputs, true_keypointsV)*100)

                    val_loss += loss.item()

                    true_keypointsV = denormalize_keypoints(true_keypointsV, W, H)
                    outputs = denormalize_keypoints(outputs, W, H)
                   
                    distanze_batch = torch.norm(outputs - true_keypointsV, dim=-1)

                    # Soglia utilizzata per il calcolo della SDR
                    soglia = 2  
                    punteggi_batch = (distanze_batch <= soglia).float()
                    
                    # Array di distanza per ogni batch
                    distanze_totali.append(punteggi_batch)


                    val_mre += calcola_mre(outputs,true_keypointsV)
                    
                    val_sdr += calcola_sdr(outputs,true_keypointsV,soglia)
                    
            # Concatena le distanze di tutti i batch
            distanze_totali = torch.cat(distanze_totali, dim=0)

            # Calcola la metrica media per ciascun keypoint su tutti i dati
            metriche_per_keypoint = distanze_totali.mean(dim=0).cpu().numpy() *100
            
            # Stampa la metrica media per ciascun keypoint
            print(f"\nPrecisione per ogni keypoint con soglia pari a {soglia} pixel: ")
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

            val_med_sdr = val_sdr / len(val_loader)
            print(f'***** Metrica SDR: {val_med_sdr:.5f} *****\n\n')

            val_losses.append(val_loss_avg)
            metric_mre.append(val_med_mre)
            metric_sdr.append(val_med_sdr)

            # Controllo per early stopping e per tenere i pesi del modello più basso di loss
            if val_loss_avg <= best_val_loss:
                best_val_loss = val_loss_avg
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve +=1
                if epochs_no_improve == patience:
                    early_stop = True
                    break

    scheduler.step(val_loss_avg)
        

    stampa_mappa(hm,inputs, 0)
    stampa_mappa(true_heatmapsV,inputs, 1)

    stampa_metric_key(metric_key)
    # Tracciamento delle loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss During Training and Validation')
    plt.legend()
    plt.savefig('graficoTrain.png')
    plt.close()

    # Tracciamento mre
    plt.figure(figsize=(10, 5))
    plt.plot(metric_mre, label = 'mre')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.title('Metric Mean Relative Error')
    plt.legend()
    plt.savefig('graficoMse.png')
    plt.close()

    # Tracciamento sdr
    plt.figure(figsize=(10, 5))
    plt.plot(metric_sdr, label = 'sdr')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.title('Metric Success Detection Rate')
    plt.legend()
    plt.savefig('graficoSdr.png')
    plt.close()

    if early_stop:
        print("\nAddestramento interrotto a causa dell'early stopping")
    else:
        print("\nAddestramento completato")

    if best_model_state:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), 'best_saved_model_TransKeyR.pth')



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

    plt.savefig('TestPrediction.png')
    plt.close()


if __name__ == "__main__":
    main()
