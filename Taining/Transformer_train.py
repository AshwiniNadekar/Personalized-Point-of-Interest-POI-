from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from node2vec import Node2Vec
import torch.nn as nn
import torch.nn.functional as F
import torch
import math 
from Models.transformer_model import TransformerModule
import pickle


class POIDataset(Dataset):
    def __init__(self, csv_file, seq_len=5, hop=1):
        self.seq_len = seq_len
        self.hop = hop
        self.data = pd.read_csv(csv_file)
        self.data=self.data[self.data['user'].isin(self.data['user'].unique()[:1000])]
        self.data.dropna(inplace=True)
        self.data['check-in time']=pd.to_datetime(self.data['check-in time'], utc=True)
        self.data['month']=self.data['check-in time'].dt.month
        self.data['day']=self.data['check-in time'].dt.day
        self.data['hour']=self.data['check-in time'].dt.hour
        self.data['minute']=self.data['check-in time'].dt.minute
        self.data['seconds'] = self.data['check-in time'].dt.second
        self.data.sort_values(by=['user','check-in time'], inplace=True)
        self.data.reset_index(inplace=True,drop=True)
        self.indices = []
        with open(os.path.join("Dataset","node2vec_model.pkl"), "rb") as f:
            self.embedding_model = pickle.load(f)
        for idx in range(0,len(self.data),hop):
            if idx+self.seq_len >= len(self.data):
                break

            if self.data.loc[idx+seq_len-1,'user']==self.data.loc[idx,'user']:
                self.indices.append(idx)

        self.indices = np.array(self.indices)
        print(self.data['longitude'].dtype,self.data['latitude'].dtype)

        print(self.indices)
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        user=str(self.data['user'][self.indices[idx]])
        user_embedding=torch.tensor(self.embedding_model.wv[user]).unsqueeze(0).repeat(self.seq_len,1)
        input_seq = torch.tensor(self.data.loc[self.indices[idx]:self.indices[idx]+self.seq_len-1,['month','day','hour','minute','seconds','latitude','longitude']].values, dtype=torch.float32)
        input_seq = torch.cat([user_embedding,input_seq],dim=1)
        target=torch.tensor(self.data.loc[self.indices[idx]+self.seq_len:self.indices[idx]+self.seq_len,['latitude','longitude']].values, dtype=torch.float32).unsqueeze(0)
        
        return (input_seq, target)

train_dataset=POIDataset(os.path.join("Dataset","train_data.csv"),seq_len=10,hop=4)
train_loader=DataLoader(train_dataset,batch_size=32,drop_last=True)

    
device = "cuda" if torch.cuda.is_available() else "cpu"
model=TransformerModule(64,8,6,10,71,device)
model.to(device)
criterion=torch.nn.MSELoss()
epochs=15
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
for ep in range(epochs):
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training {ep+1}/{epochs}"):
        # Get inputs and targets
        sequence, target = batch[0].to(device), batch[1].to(device).reshape(32,2)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(sequence)
        # Compute loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track loss
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{ep+1}/{epochs}] - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(),f"POI_Transformer.pth")