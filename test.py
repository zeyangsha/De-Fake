from time import process_time_ns
import torch
import clip
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
 
from sklearn.metrics import confusion_matrix
import itertools
import torch.nn.functional as F

from clipdatasets import real,fakereal,realflickr
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
from torch import nn
import sys
import argparse
import time
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve
from models.blip import blip_decoder

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

def preprocess_image(img_path, image_size=224):
    img = Image.open(img_path)
    img = img.resize((image_size, image_size))
    return preprocess(img)

parser = argparse.ArgumentParser(description='Finetune the classifier to wash the backdoor')
parser.add_argument('--image_path',default='CLIP.png',type=str)
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-B/32")

image_size = 224

image = preprocess_image(args.image_path,image_size).unsqueeze(0).to(device)

blip_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

blip = blip_decoder(pretrained=blip_url, image_size=image_size, vit='base')
blip.eval()
blip = blip.to(device)

caption = blip.generate(image, sample=False, num_beams=3, max_length=60, min_length=5) 
text = clip.tokenize(list(caption)).to(device)

model = torch.load("/home/c01zesh/CISPA-projects/fake_artist-2022/clip/CLIP/finetune_clip.pt").to(device)

linear = NeuralNet(1024,[512,256],2).to(device)
linear = torch.load('/home/c01zesh/CISPA-projects/fake_artist-2022/clip/CLIP/clip_linear.pt')

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    emb = torch.cat((image_features, text_features),1)
    output = linear(emb.float())
    predict = output.argmax(1)
    predict = predict.cpu().numpy()
    print(predict)
