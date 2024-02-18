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

from clipdatasets import real,fakereal
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
from torch import nn
from log import get_logger
import sys
import argparse
import time
from tqdm import tqdm


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

temp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
print(temp_time)
print(os.path.basename(sys.argv[0]).split('.')[0])
if not os.path.isdir(your_dir):
    os.mkdir(your_dir)

logger = get_logger(your_dir_log)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
size = 224
train_dataset_0 = real('your data',size)
train_dataset_1 = fakereal('your data',size)
train_dataset_2 = fakereal('your data',size)
size_for_this = 10000
train_dataset_0,_ =  random_split(dataset=train_dataset_0,lengths=[size_for_this,len(train_dataset_0)-size_for_this],generator=torch.Generator().manual_seed(0))
train_dataset_1,_ =  random_split(dataset=train_dataset_1,lengths=[size_for_this,len(train_dataset_1)-size_for_this],generator=torch.Generator().manual_seed(0))
train_dataset_2,_ =  random_split(dataset=train_dataset_2,lengths=[size_for_this,len(train_dataset_2)-size_for_this],generator=torch.Generator().manual_seed(0))
train_dataset = torch.utils.data.ConcatDataset([train_dataset_0,train_dataset_1,train_dataset_2])
size = len(train_dataset)
newsize = size*0.8
    
train_dataset,test_dataset = random_split(dataset=train_dataset,lengths=[int(newsize),int(size-newsize) ],generator=torch.Generator().manual_seed(0))
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )

test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )
linear = NeuralNet(1024,[512,256],2).to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(linear.parameters())+list(model.parameters()), lr=3e-4)


for i in range(100):
    loss_epoch = 0
    train_acc = []
    train_true = []
    
    test_acc = []
    test_true = []

    for step, (x,y,t) in enumerate(tqdm(train_loader)):
        x = x.cuda()
        y = y.cuda()
        linear.train()
        text = clip.tokenize(list(t)).to(device)
        with torch.no_grad():
            imga_embedding = model.encode_image(x)
            text_emb = model.encode_text(text)
        emb = torch.cat((imga_embedding,text_emb),1)
        output = linear(emb.float())
        optimizer.zero_grad()
        loss = criterion(output,y)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        predict = output.argmax(1)
        predict = predict.cpu().numpy()
        predict = list(predict)
        train_acc.extend(predict)
        
        y = y.cpu().numpy()
        y = list(y)
        train_true.extend(y)
        
    for step, (x,y,t) in enumerate(tqdm(test_loader)):
        x = x.cuda()
        y = y.cuda()
        model.eval()
        linear.eval()
        text = clip.tokenize(list(t)).to(device)
        with torch.no_grad():
            imga_embedding = model.encode_image(x)
            text_emb = model.encode_text(text)
    
        emb = torch.cat((imga_embedding,text_emb),1)
        emb = imga_embedding
        output = linear(emb.float())
        predict = output.argmax(1)
        predict = predict.cpu().numpy()
        predict = list(predict)
        test_acc.extend(predict)
        
        y = y.cpu().numpy()
        y = list(y)
        test_true.extend(y)
    
    print('train')
    print(accuracy_score(train_true,train_acc))
    logger.info('Epoch:[{}/{}]\t loss={:.5f}\t train acc={:.5f}'.format(str(i), str(
        100), loss_epoch, accuracy_score(train_true,train_acc)))
    print('test')
    print(accuracy_score(test_true,test_acc))
    logger.info('Test\t clean_acc={:.5f}'.format(accuracy_score(test_true,test_acc)))  
