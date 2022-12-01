from tqdm import tqdm
import os
import shutil
import zipfile
import tarfile
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
import cv2
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
from model_components import *



"""Dataset Class for Audiovisual data from Voxceleb1"""
class IdentificationDataset(Dataset):
    
    def __init__(self, path, train, audio_transform=None, video_transform=None):
        #dataset split from data download site
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
        #path like id10003/L9_sh8msGV59/00001.wav
        track_path = self.dataset[idx]
        #path to audio data
        audio_path = os.path.join(self.path, 'audio', track_path)
        #path to video data
        video_path = os.path.join(self.path, 'video', '/'.join(track_path.split('/')[:-1])) + '/'
        #path to URL and timestamp metadata
        txt_path = video_path + track_path.split('/')[-1].replace('.wav','')+'.txt'

        if os.path.exists(audio_path) and os.path.exists(txt_path):
           
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
                segment_len = 2 # sec
                upper_bound = len(samples) - segment_len * rate
                start = np.random.randint(0, upper_bound)
                end = start + segment_len * rate
                samples = samples[start:end]

            # spectogram spec = freq x time
            
            _, _, spec = signal.spectrogram(samples, rate, window, Nw, Ns, nfft, 
                                            mode='magnitude', return_onesided=True)
#             print('samples {}'.format(samples.shape))


            # multiply by 1600
            spec *= rate / 10

            #apply normalization
            if self.audio_transform:
                spec = self.audio_transform(spec)
                
            #identify video frames corresponding to wav file    
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
                frames = torch.stack([self.video_transform(cv2.resize(img, (60,60))) for img in imgs])
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
EPOCH_NUM = 30

torch.backends.cudnn.deterministic = True
#batch size
B = 100
#hyperparameters (need to tune)
WEIGHT_DECAY = 5e-4
LR_INIT = 1e-2
LR_LAST = 1e-4
# lr scheduler parameter
gamma = 10 ** (np.log10(LR_LAST / LR_INIT) / (EPOCH_NUM - 1))
MOMENTUM = 0.9
DEVICE = 'cuda:0'
NUM_WORKERS = 1


audio_transforms = Compose([
    Normalize(),
    ToTensor()
])
import torchvision.transforms
video_transform = torchvision.transforms.ToTensor()

#custom collate function for dataloader because data is variable size
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

#create pytorch dataloader objects
trainset = IdentificationDataset(DATASET_PATH, train=True, audio_transform=audio_transforms, video_transform=video_transform)
trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=B, num_workers=NUM_WORKERS, collate_fn=collate_fn,shuffle=True)

testset = IdentificationDataset(DATASET_PATH, train=False, audio_transform=audio_transforms, video_transform=video_transform)
testsetloader = torch.utils.data.DataLoader(testset, batch_size=1, collate_fn=collate_fn,num_workers=NUM_WORKERS*2)

criterion = nn.CrossEntropyLoss()


def euclidean_dis(x, reconstructed_x):
	dis = torch.dist(x,reconstructed_x,2)
	return dis

def avgpooling(x):
	m = nn.AvgPool2d(7)
	return m(x)

#compute loss between input and vae reconstruction
def caluculate_loss_generaldec(x_visual, x_audio, x_reconstruct, mu, logvar, epoch, video_frames, reconstruction):
    loss_MSE = nn.MSELoss()
    mse_loss = 0
    video_idx = 0
    audio_idx = 0
    #need to calculate loss for each sample because 1 audio tensor corresponds to a variable number of visual tensors
    if reconstruction == 'video':
        for frame in video_frames:
            no_of_frames = frame.size(0)
            for f in range(no_of_frames):
                x_input = torch.cat([x_visual[video_idx,:], x_audio[audio_idx,:]], dim = 0)
                mse_loss += loss_MSE(x_input,x_reconstruct[video_idx,:])
                video_idx += 1
            audio_idx += 1
    else:
        for frame in video_frames:
            no_of_frames = frame.size(0)
            for f in range(no_of_frames):
                x_input = torch.cat([x_visual[video_idx,:], x_audio[audio_idx,:]], dim = 0)
                mse_loss += loss_MSE(x_input,x_reconstruct[audio_idx,:])
                video_idx += 1
            audio_idx += 1

    #KL divergence component of loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    _ , mu1, logvar1 = vae_audio(x_audio)
    z1 = vae_audio.reparameterize(mu1, logvar1)
    _, mu2, logvar2 = vae_video(x_visual)
    z2 = vae_video.reparameterize(mu2, logvar2)
    
    #compute Wasserstein latent loss
    latent_loss = 0
    video_idx = 0
    audio_idx = 0
    for frame in video_frames:
        no_of_frames = frame.size(0)
        loss = 0
        for f in range(no_of_frames):
            loss += euclidean_dis(z1[audio_idx,:], z2[video_idx,:])
            video_idx += 1
        audio_idx += 1
        latent_loss += loss/no_of_frames
    
    if epoch < 5:
        final_loss = mse_loss + kl_loss*0.1 + latent_loss
    else:
        final_loss = mse_loss + kl_loss*0.01 + latent_loss
        
    return final_loss, kl_loss, mse_loss, latent_loss

"""Main TRAIN funcion"""
def train_generaldec(epoch):
    vae_audio.train()
    vae_video.train()
    train_loss = 0
    kl_loss = 0
    mse_loss = 0
    latent_loss = 0
    training_size = len(trainsetloader)
    for idx, sample in enumerate(trainsetloader):
        #resize/pool data
        
        #[total no of frames in batch, 3, 23, 23]
        visual_data_input = torch.cat(sample['video_frames'], dim = 0).cuda()
        #[total no of frames in batch, 23, 23]
        visual_data_input = torch.mean(visual_data_input, dim = 1)
         #[total no of frames in batch, 529]
        visual_data_input = torch.flatten(visual_data_input, start_dim = 1, end_dim = 2)
        #[no of samples in batch, 257,198]
        audio_data_gt = torch.cat(sample['audio'], 0).cuda()

        #average freqs across time to get [no of samples in batch, 257]
        audio_data_gt = torch.mean(audio_data_gt, dim = 2)
        optimizer_audio.zero_grad()
        optimizer_video.zero_grad()
        if epoch == 0:
            x_reconstruct_from_v, mu1, logvar1 = vae_video(visual_data_input)
        else:
            x_reconstruct_from_v, mu1, logvar1 = vae_video(visual_data_input,vae_audio)
        #compute loss terms

        loss1, kl1, mse1, latent1 = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct_from_v, mu1, logvar1, epoch, sample['video_frames'], 'video')

        x_reconstruct_from_a, mu2, logvar2 = vae_audio(audio_data_gt, vae_video)
        loss2, kl2, mse2, latent2 = caluculate_loss_generaldec(visual_data_input, audio_data_gt, x_reconstruct_from_a, mu2, logvar2, epoch, sample['video_frames'], 'audio')

        loss = loss1 + loss2
        kl = kl1 + kl2
        mse = mse1 + mse2
        latent = latent1 + latent2

        loss.backward()
        train_loss += loss.item()
        kl_loss += kl.item()
        mse_loss += mse.item()
        latent_loss += mse.item()

        optimizer_video.step()
        optimizer_audio.step()
    return train_loss, kl_loss, mse_loss, latent_loss


latent_dim = 100
epoch_nb = 8

audio_encode = audio_encoder(latent_dim)
video_encode = visual_encoder(latent_dim)
general_decode = general_decoder(latent_dim)
vae_audio = VAE(latent_dim, audio_encode, general_decode)
vae_video = VAE(latent_dim, video_encode, general_decode)

vae_audio.cuda()
vae_video.cuda()

optimizer_audio = optim.Adam(vae_audio.parameters(), lr = 0.00001)
optimizer_video = optim.Adam(vae_video.parameters(), lr = 0.00001)

#main train loop
for epoch in range(epoch_nb):
    train_loss = 0
    train_loss, kl_loss, mse_loss, latent_loss = train_generaldec(epoch)
    train_loss /= training_size
    kl_loss /= training_size
    mse_loss /= training_size
    latent_loss /= training_size
    print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')
#save model weights
torch.save(vae_audio.state_dict(), '8_msvae_a.pkl')
torch.save(vae_video.state_dict(), '8_msvae_v.pkl')