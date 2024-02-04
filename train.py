import torch
import os
import librosa
import numpy as np
import pandas as pd
import torch.backends.cudnn
import random
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
import wave
import torch.nn as nn
import torch.nn.functional as F
import time
from .datasets import VoiceDataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from transformers import AutoModelForAudioClassification, WhisperProcessor
from datasets import load_dataset

def main(args):
    seed_num = 42
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)

    np.random.seed(seed_num)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed_num)
    ckpt='openai/whisper-large'
    processor = WhisperProcessor.from_pretrained(ckpt, language="Korean")
    model = AutoModelForAudioClassification.from_pretrained(ckpt)
    model.config.forced_decoder_ids = None

    root_path = '/content/drive/My Drive/VoiceTrain/Voice'
    train_dataset = VoiceDataset(root_path)
    
    batch_size = 32
    sampling_rate = 16000
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    epoch=3
    batch_size = 32
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for i in range(epoch):
        
        model.train()
        for iter, data in enumerate(train_dataloader):
            train_losses = []
            x = data[0].to(device)
            x_labels = data[1].to(device)
            
            optimizer.zero_grad()
            logit = model(x)[0]

            loss = criterion(logit, x_labels)
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()


        if (iter+1) % 50 == 0:
            print(f"{i+1}/{iter+1} train loss : {np.array(train_losses).mean()}")
            train_losses = []