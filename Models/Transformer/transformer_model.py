import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class SinEmbedding(nn.Module):

    def __init__(self,d_model,max_len=10,device="cpu"):
        super(SinEmbedding,self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.device=device
        self.positional_embedding = torch.zeros(max_len,d_model) #(max_num_tokens,d_feature_embeddings)
        position = torch.arange(0,max_len).unsqueeze(1) #(max_len,1)
        den_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000))/d_model).unsqueeze(0)#(1,dim_model)

        self.positional_embedding[:,0::2] = torch.sin(position*den_term)
        self.positional_embedding[:,1::2] = torch.cos(position*den_term)

    def forward(self,x):
        #x: (batch,max_len,dim)
        return x + self.positional_embedding.unsqueeze(0).to(device)
    

class TransformerModule(nn.Module):

    def __init__(self,d_model,nhead,num_layers,max_len=10,input_dim=64,device="cpu"):
        super(TransformerModule,self).__init__()
    
        self.device=device
        self.max_len=max_len
        self.input_dim=input_dim
        self.conv1d=nn.Conv1d(input_dim,d_model,kernel_size=3,padding=1,stride=1)
        self.PositionalEmbedding = SinEmbedding(d_model,max_len,device=device)
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead)
        self.transformerEncoder = nn.TransformerEncoder(self.encoderLayer,num_layers=num_layers)
        self.regressor = nn.Linear(d_model,2)

        self.cls = nn.Parameter(torch.zeros(1,1,d_model))


    def forward(self,x):

        #x:(batch_size,seq_length,x_shape)

        
        self.cls_embedding = self.cls.expand(x.shape[0],1,-1)
        #(batch_size,seq_length,x_shape) - (batch_size,x_shape,seq_length)
        x=x.transpose(-1,-2) 
        #(batch_size,x_shape,seq_length) -(batch_size,d_model,seq_length)
        x=self.conv1d(x)
        #(batch_size,d_model,seq_length) - #(batch_size,seq_length,d_model)
        x=x.transpose(-1,-2)
        #(batch_size,seq_length,d_model)
        x=self.PositionalEmbedding(x)
        x=torch.cat((self.cls_embedding,x),dim=1)
        #(batch_size,seq_length,d_model)- (seq_length,batch_size,d_model)
        x=x.transpose(0,1)
        #(seq_length,batch_size,d_model) same output
        x=self.transformerEncoder(x)
        x=x[0,:,:] # taking only cls token # (1,batch_size,d_model)
        x=x.squeeze(0)
        x=self.regressor(x)

        return x




