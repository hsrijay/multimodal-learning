import os

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import tensorboardX
from tqdm import tqdm

import matplotlib.pyplot as plt


class IdentificationDataset(Dataset):
    
    def __init__(self, path, train, transform=None):
        iden_split_path = os.path.join(path, 'iden_split.txt')
        split = pd.read_table(iden_split_path, sep=' ', header=None, names=['phase', 'path'])
        
        if train:
            phases = [1, 2]
        
        else:
            phases = [3]
            
        mask = split['phase'].isin(phases)
        self.dataset = split['path'][mask].reset_index(drop=True)
        self.path = path
        self.train = train
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # path
        track_path = self.dataset[idx]
        audio_path = os.path.join(self.path, 'audio', track_path)

        # read .wav
        rate, samples = wavfile.read(audio_path)
        # extract label from path like id10003/L9_sh8msGV59/00001.txt
        # subtracting 1 because PyTorch assumes that C_i in [0, 1251-1]
        label = int(track_path.split('/')[0].replace('id1', '')) - 1

        ## parameters
        window = 'hamming'
        # window width and step size
        Tw = 25 # ms
        Ts = 10 # ms
        # frame duration (samples)
        Nw = int(rate * Tw * 1e-3)
        Ns = int(rate * (Tw - Ts) * 1e-3)
        # overlapped duration (samples)
        # 2 ** to the next pow of 2 of (Nw - 1)
        nfft = 2 ** (Nw - 1).bit_length()
        pre_emphasis = 0.97
        
        # preemphasis filter
        samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])
        
        # removes DC component of the signal and add a small dither
        samples = signal.lfilter([1, -1], [1, -0.99], samples)
        dither = np.random.uniform(-1, 1, samples.shape)
        spow = np.std(samples)
        samples = samples + 1e-6 * spow * dither
        
        if self.train:
            # segment selection
            segment_len = 3 # sec
            upper_bound = len(samples) - segment_len * rate
            start = np.random.randint(0, upper_bound)
            end = start + segment_len * rate
            samples = samples[start:end]
        
        # spectogram
        _, _, spec = signal.spectrogram(samples, rate, window, Nw, Ns, nfft, 
                                        mode='magnitude', return_onesided=False)
        
        # just multiplying it by 1600 makes spectrograms in the paper and here "the same"
        spec *= rate / 10
        
        if self.transform:
            spec = self.transform(spec)

        return label, spec
    
class Normalize(object):
    """Normalizes voice spectrogram (mean-varience)"""
    
    def __call__(self, spec):
        
        # (Freq, Time)
        # mean-variance normalization for every spectrogram (not batch-wise)
        mu = spec.mean(axis=1).reshape(512, 1)
        sigma = spec.std(axis=1).reshape(512, 1)
        spec = (spec - mu) / sigma

        return spec

class ToTensor(object):
    """Convert spectogram to Tensor."""
    
    def __call__(self, spec):
        F, T = spec.shape
        
        # now specs are of size (Freq, Time) and 2D but has to be 3D (channel dim)
        spec = spec.reshape(1, F, T)
        
        # make the ndarray to be of a proper type (was float64)
        spec = spec.astype(np.float32)
        
        return torch.from_numpy(spec)
    
    
DATASET_PATH = '/home/nvme/data/vc1/'
LOG_PATH = '/home/nvme/logs/VoxCeleb/rm_dc_n_dither'
EPOCH_NUM = 30

# in shared code B = 100 but PyTorch throws CUDA out of memory at B = 97 
# though B=96 takes only 90.6% of the GPU Mem (bug?):
# https://discuss.pytorch.org/t/lesser-memory-consumption-with-a-larger-batch-in-multi-gpu-setup/29087
# B = 96
# but when 
torch.backends.cudnn.deterministic = True
# I can set B = 100
B = 100

WEIGHT_DECAY = 5e-4
LR_INIT = 1e-2
LR_LAST = 1e-4
# lr scheduler parameter
gamma = 10 ** (np.log10(LR_LAST / LR_INIT) / (EPOCH_NUM - 1))
MOMENTUM = 0.9
DEVICE = 'cuda:0'
NUM_WORKERS = 4
TBoard = tensorboardX.SummaryWriter(log_dir=LOG_PATH)

net = VoiceNet(num_classes=1251)
net.to(DEVICE)

transforms = Compose([
    Normalize(),
    ToTensor()
])

trainset = IdentificationDataset(DATASET_PATH, train=True, transform=transforms)
trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=B, num_workers=NUM_WORKERS, shuffle=True)

testset = IdentificationDataset(DATASET_PATH, train=False, transform=transforms)
testsetloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=NUM_WORKERS*2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), LR_INIT, MOMENTUM, weight_decay=WEIGHT_DECAY)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)