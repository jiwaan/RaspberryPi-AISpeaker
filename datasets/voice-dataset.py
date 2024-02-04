import torch
import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
import wave
import datasets
from torch.utils.data import Dataset



class VoiceDataset(Dataset):
  def __init__(self, root_path, split='train'):
    sampling_rate = 16000
    data = []
    label = []
    aa = pd.read_json(os.path.join(root_path + '/voice.json'))    
    for f in [(file_name, label) for file_name, label in zip(list(aa.file), list(aa.labeling))]:
      data.append(librosa.load((os.path.join(root_path + '/' + f[0])), sr=sampling_rate)[0]) #array
      label.append(f[-1])

    if data is not None:
      data = processor(data, sampling_rate=sampling_rate, return_tensor='pt').input_features

    self.data = data
    self.label = label
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i], self.label[i]