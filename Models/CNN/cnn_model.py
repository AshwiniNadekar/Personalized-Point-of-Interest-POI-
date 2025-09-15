import torch.nn as nn
import torch.nn.functional as F
import torch
import math
class CNNModel(nn.Module):

    def __init__(self,dim_loc,max_len=10):


        self.conv_sequence=nn.Sequential(*[(nn.Conv1d(dim_loc,3,2,padding='same'),nn.Conv1d(3,3,2,padding='same'),nn.Conv1d(3,3,2,padding='same')
                                            ,nn.Conv1d(3,3,2,padding='same'),nn.BatchNorm1d(3),nn.Relu(),nn.Conv1d(3,16,2,padding='same'),nn.Conv1d(16,32,2,1),nn.Conv1d(32,32,2,1)
                                            ,nn.BatchNorm1d(32),nn.Relu(),nn.Conv1d(32,64,2,1),nn.Conv1d(64,128,2,1),nn.Conv1d(128,128,2,1),nn.Conv1d(128,128,2,1))])
        
        self.dropout=nn.Dropout(0.01)
        self.flatten=nn.Flatten()
        self.ffn=nn.Sequential(*[nn.Linear(1280,4096),nn.Relu(),nn.Linear(4096,2)])
    def forward(self,inp):

        #inp: (BS,len,dim_loc)
        
        inp=inp.transpose(1,2)
        #inp: (BS,dim_loc,len)

        for layer in self.conv_sequence:
            inp=layer(inp)

        #inp: (BS,128,max_len)
        inp=self.flatten(inp)
        #inp: (BS,128*max_len)
        inp=self.ffn(inp)
        #inp:(BS,1,2)
        return inp

        