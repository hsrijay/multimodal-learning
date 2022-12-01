#imports
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import torchvision
from torchvision.transforms import Compose
import torch.optim as optim
import math
import torch.utils.data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from model_components import *
import pickle
from torch.utils.data import Dataset
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import cv2
from skimage import io


#dataset class
class IdentificationDataset(Dataset):
    
    def __init__(self, path, train, audio_transform=None, video_transform=None):
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
        self.audio_transform = audio_transform
        self.video_transform = video_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
#         print('start\n')
        # path
        track_path = self.dataset[idx]
        audio_path = os.path.join(self.path, 'audio', track_path)
        video_path = os.path.join(self.path, 'video', '/'.join(track_path.split('/')[:-1])) + '/'
        txt_path = video_path + track_path.split('/')[-1].replace('.wav','')+'.txt'

        if os.path.exists(audio_path) and os.path.exists(txt_path):
           
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
                segment_len = 2 # sec
                upper_bound = len(samples) - segment_len * rate
                start = np.random.randint(0, upper_bound)
                end = start + segment_len * rate
                samples = samples[start:end]

            # spectogram spec = freq x time
            
            _, _, spec = signal.spectrogram(samples, rate, window, Nw, Ns, nfft, 
                                            mode='magnitude', return_onesided=True)


            # just multiplying it by 1600 makes spectrograms in the paper and here "the same"
            spec *= rate / 10

            if self.audio_transform:
                spec = self.audio_transform(spec)
                
            #identify corresponding video frames to wav file    
            all_frames = dict(zip([x.lstrip("0") for x in os.listdir(video_path)],[x for x in os.listdir(video_path)]))
            #get corresponding frames from corresponding txt file
            f=open(txt_path,"r")
            lines=f.readlines()
            result=[]
            for x in lines:
                result.append(x.split(' ')[0])
            f.close()
            result = result[7:]
            #collect corresponding frames
            images_in_tensor = []
            for frame in result:
                if frame.lstrip("0") + '.jpg' in all_frames:
                    images_in_tensor.append(frame.lstrip("0") + '.jpg')
            imgs = io.imread_collection([video_path + all_frames[x] for x in images_in_tensor], conserve_memory = True)
            #convert to tensor and resize frames to 23 x 23
            if len(imgs) > 0:
                frames = torch.stack([self.video_transform(cv2.resize(img, (23,23))) for img in imgs])
            else:
                return
                
            

            return label, spec, frames
        else:
            return
       
    
class Normalize(object):
    """Normalizes voice spectrogram (mean-varience)"""
    
    def __call__(self, spec):
        
        # (Freq, Time)
        # mean-variance normalization for every spectrogram (not batch-wise)
        mu = spec.mean(axis=1).reshape(257, 1)
        sigma = spec.std(axis=1).reshape(257, 1)
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


        
DATASET_PATH = '/usr/xtmp/hs285/'
NUM_WORKERS = 1
B = 100


audio_transforms = Compose([
    Normalize(),
    ToTensor()
])
import torchvision.transforms
from skimage import io

video_transform = torchvision.transforms.ToTensor()

def collate_fn(batch):
    batch = list(filter(lambda x : x is not None, batch))
    label = []
    audio = []
    video = []
    for item in batch:
        label.append(item[0])
        audio.append(item[1])
        video.append(item[2])
    return {'label': label,
           'audio': audio,
           'video_frames': video}

trainset = IdentificationDataset(DATASET_PATH, train=True, audio_transform=audio_transforms, video_transform=video_transform)
trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=B, num_workers=NUM_WORKERS, collate_fn=collate_fn,shuffle=True)

testset = IdentificationDataset(DATASET_PATH, train=False, audio_transform=audio_transforms, video_transform=video_transform)
testsetloader = torch.utils.data.DataLoader(testset, batch_size=1, collate_fn=collate_fn,num_workers=NUM_WORKERS*2)
#preprocessed audio and video data
x_a = []
x_v = []
labels= []
#latent representations of audio and video learnt by vae
z_a = []
z_v = []
latent_dim = 100


audio_encode = audio_encoder(latent_dim)
audio_decode = general_decoder(latent_dim)
video_encode = visual_encoder(latent_dim)
video_decode = general_decoder(latent_dim)
vae_audio = VAE(latent_dim, audio_encode, audio_decode)
vae_video = VAE(latent_dim, video_encode, video_decode)
#load from trained model
vae_audio.load_state_dict(torch.load('8_msvae_a.pkl'))
vae_video.load_state_dict(torch.load('8_msvae_v.pkl'))

#generate latent multi-modal representations learnt by VAE for each sample in test set
vae_audio.eval()
vae_video.eval()
for idx, sample in enumerate(testsetloader):
    print(idx)
    v = sample['video_frames']
    a = sample['audio']
    if len(v)>0 and len(a)>0:
        v = torch.flatten(torch.mean(v[0], dim = 1), start_dim = 1, end_dim = 2)
        a = torch.mean(a[0],dim = 2)
        label = sample['label']
        v = torch.unsqueeze(v[0],0)
        x_a.append(torch.squeeze(a).numpy())
        x_v.append(torch.squeeze(v).numpy())
        labels.append(label[0])
        vae_video.eval()
        vae_audio.eval()
        #generate encoded representations by vae
        with torch.no_grad():
            mu1, logvar1 = vae_video.encoder(v)
            mu2, logvar2 = vae_audio.encoder(a)
        z_v.append(torch.squeeze(mu1).numpy())
        z_a.append(torch.squeeze(mu2).numpy())
        
#write data to file
with open("visualize", "wb") as fp:   
    pickle.dump(x_a, fp)
    pickle.dump(x_v, fp)
    pickle.dump(z_a, fp)
    pickle.dump(z_v, fp)
    pickle.dump(labels, fp)