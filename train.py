import os
import sys
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from time import process_time_ns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split
from natsort import natsorted
import torch.nn.functional as F
import clip


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

# 定义数据集类
class real(Dataset):
    def __init__(self, root_dir1, prompts_file, transform=None):
        self.root_dir1 = root_dir1
        self.transform = transform
        self.image_filenames1 = natsorted(os.listdir(root_dir1))
        self.prompts = self.load_prompts(prompts_file)

    def load_prompts(self, prompts_file):
        with open(prompts_file, 'r') as file:
            prompts = file.readlines()
        prompts = [prompt.strip() for prompt in prompts]  # Remove any extra whitespace
        return prompts

    def __len__(self):
        return len(self.image_filenames1)

    def __getitem__(self, idx):
        class_name1 = self.image_filenames1[idx]
        image_path1 = os.path.join(self.root_dir1, class_name1)
        image1 = Image.open(image_path1).convert("RGB")
        
        if self.transform:
            image1 = self.transform(image1)
        
        label = 0
        prompt = self.prompts[idx] if idx < len(self.prompts) else ""

        return image1, prompt, label

class fake(Dataset):
    def __init__(self, root_dir1, prompts_file, transform=None):
        self.root_dir1 = root_dir1
        self.transform = transform
        self.image_filenames1 = natsorted(os.listdir(root_dir1))
        self.prompts = self.load_prompts(prompts_file)

    def load_prompts(self, prompts_file):
        with open(prompts_file, 'r') as file:
            prompts = file.readlines()
        prompts = [prompt.strip() for prompt in prompts]
        return prompts

    def __len__(self):
        return len(self.image_filenames1)

    def __getitem__(self, idx):
        class_name1 = self.image_filenames1[idx]
        image_path1 = os.path.join(self.root_dir1, class_name1)
        image1 = Image.open(image_path1).convert("RGB")
        
        if self.transform:
            image1 = self.transform(image1)
        
        label = 1
        prompt = self.prompts[idx] if idx < len(self.prompts) else ""

        return image1, prompt, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

realdata = real(root_dir1="real-data",prompts_file="prompt.txt", transform=transform)
fakedata = fake(root_dir1="fake-data",prompts_file="prompt.txt", transform=transform)

dataset = torch.utils.data.ConcatDataset([realdata,fakedata])

newsize = 800

size = len(dataset)
train_dataset,test_dataset = random_split(dataset=dataset,lengths=[int(newsize),int(size-newsize)],generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
linear = NeuralNet(1024,[512,256],2).to(device)
model.to(device)
optimizer = torch.optim.Adam(list(linear.parameters())+list(model.parameters()), lr=3e-4)
criterion = nn.CrossEntropyLoss()
linear.to(device)

for epoch in range(50):
    model.train()
    linear.train()
    for batch_idx, (data1, prompt, target) in enumerate(train_loader):
        data1, target = data1.to(device), target.to(device)
        text = clip.tokenize(list(prompt)).to(device)
        with torch.no_grad():
            imga_embedding = model.encode_image(data1)
            text_emb = model.encode_text(text)
        emb = torch.cat((imga_embedding,text_emb),1)
        output = linear(emb.float())
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0
    model.eval()
    linear.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, prompts, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            text_tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                image_embeddings = model.encode_image(images)
                text_embeddings = model.encode_text(text_tokens)

            embeddings = torch.cat((image_embeddings, text_embeddings), 1)
            outputs = linear(embeddings.float())
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, average='weighted')
    precision = precision_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}')

torch.save(model.state_dict(), 'train_clip_model.pth')
torch.save(linear.state_dict(), 'train_linear_model.pth')
