import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
import wave

import torch.nn as nn
import torch.nn.functional as F
import time
import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from transformers import AutoModelForAudioClassification, WhisperProcessor



class Inferencer:
  def __init__(self, device='cuda'):
    if device == 'cuda':
      assert torch.cuda.is_available()
    self.device = device
		
		# model 불러오기
    model_name = 'openai/whisper-large'
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    ckp_file = "whisper_finetuned_3.pth"
    model.load_state_dict(torch.load(ckp_file, map_location=torch.device('cuda')))
    self.model = model
    self.processor = WhisperProcessor.from_pretrained(ckpt, language="Korean")
  
  # data processing
  def preprocess(self, data):
    sampling_rate = 16000
    data = librosa.load(data, sr=sampling_rate)[0]
    data = self.processor(data, sampling_rate=sampling_rate, return_tensor='pt').input_features
    data = torch.Tensor(data)
    return data
  
  def inference(self, wav_file):
    sampling_rate = 16000 #assert
    wav_file = self.preprocess(wav_file) #file_name으로 전달하면 tensor로 return

    output = self.model.forward(wav_file)
    output = output[0]
    predict_label = torch.argmax(output, dim=1)

    return predict_label