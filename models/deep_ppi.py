#-*- encoding:utf8 -*-

import os
import time
import sys

import torch as t
from torch import nn
from torch.autograd import Variable
from math import sqrt
import math


#from basic_module import BasicModule
from models.BasicModule import BasicModule

sys.path.append("../")
from utils.config import DefaultConfig
configs = DefaultConfig()

#位置编码类
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()  
        self.d_model = d_model
        self.max_len = max_len     
       
    def forward(self, x):
        pe = t.zeros(self.max_len, self.d_model)
        print(pe)
        position = t.arange(0, self.max_len, dtype=t.float).unsqueeze(1)
        print(position)
        div_term = t.exp(t.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        print(div_term)
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        pe = pe.cuda()
        return x + pe[:x.size(0), :]


#多头注意力模型类
class MultiHeadSelfAttention(nn.Module):
    dim_in:int   #input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads


    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)


    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = t.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        #零的部分全部改成非常大的负数
        mask = dist
        dist = dist.data.masked_fill(mask == 0, -1000)
        dist = t.softmax(dist, dim=-1)  # batch, nh, n, n

        att = t.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att



#局部特征卷积网络类     
class ConvsLayerL(BasicModule):
    def __init__(self):

        super(ConvsLayerL,self).__init__()
        
        self.frame_num = configs.frame_num
        self.kernels = configs.kernels[self.frame_num]
        self.pools = configs.pools[self.frame_num]
        self.stride = configs.stride[self.frame_num]
        
        in_channel = 1
        hidden_channels = configs.cnn_chanel[self.frame_num]
        
        window_size = configs.windows_size[self.frame_num]
        
        if(self.frame_num == 0):
            seq_dim = configs.seq_dim
            dssp_dim = configs.dssp_dim 
            w_size = seq_dim + dssp_dim
        if(self.frame_num == 1):
            phyc_dim = configs.phyc_dim
            ach_dim = configs.ach_dim
            w_size = phyc_dim + ach_dim
        if(self.frame_num == 2):
            pssm_dim = configs.pssm_dim
            psea_dim = configs.psea_dim
            w_size = pssm_dim + psea_dim
        
        atten_in = math.floor(((2*window_size +1) - (self.pools-1)-1)/self.stride + 1)

        atten_dim = configs.atten_dim
        atten_head = configs.atten_head

        padding = (self.kernels-1)//2
        
        #卷积模块
        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1",
            nn.Conv2d(in_channel,hidden_channels,
            padding=(padding,0),
            kernel_size=(self.kernels,w_size)))
        self.conv1.add_module("Norm1",nn.BatchNorm2d(hidden_channels))
        self.conv1.add_module("ReLU",nn.ReLU())
        self.conv1.add_module("pooling1",nn.MaxPool2d(kernel_size=(self.pools,1),stride= self.stride))

        
        #多头自注意力模块
        self.atten = nn.Sequential()
        self.atten.add_module("atten",MultiHeadSelfAttention(atten_in,atten_dim,atten_dim,atten_head))

    
    def forward(self,x):

        #输入卷积模块
        features_c = self.conv1(x)
        
        #输入注意力机制模块
        shapes_c = features_c.data.shape
        features_c = features_c.view(shapes_c[0],shapes_c[1],-1)
        
        #添加位置编码
        shapes_c = features_c.data.shape
        posEncode = PositionalEncoding(shapes_c[1],shapes_c[2])
        features_p = posEncode(features_c)
        
        #输入注意力模型
        features_a = self.atten(features_p)

        #输入深度神经网络模块
        shapes_a = features_a.data.shape
        features_a = features_a.view(shapes_a[0],shapes_a[1]*shapes_a[2])
        shapes_p = features_p.data.shape
        features_p = features_p.view(shapes_p[0],shapes_p[1]*shapes_p[2])
        features = t.cat((features_a,features_p),1)


        return features




#蛋白质相互作用位点判断网络类
class DeepPPI(BasicModule):
    def __init__(self):
        super(DeepPPI,self).__init__()
        
        #获得全局定义参数
        global configs
        self.frame_num = configs.frame_num
        self.class_num = configs.class_num
        self.dropout = configs.dropout
        
        '''
        #对原始序列做embedding，目前没有用
        seq_dim = configs.seq_dim*configs.max_sequence_length
        self.seq_layers = nn.Sequential()
        self.seq_layers.add_module("seq_embedding_layer",
        nn.Linear(seq_dim,seq_dim))
        self.seq_layers.add_module("seq_embedding_ReLU",
        nn.ReLU())
        '''

        #获得单个特征长度
        window_size = configs.windows_size[self.frame_num]
        if(self.frame_num == 0):
            seq_dim = configs.seq_dim
            dssp_dim = configs.dssp_dim            
            sigle_dim = seq_dim + dssp_dim
        if(self.frame_num == 1):
            phyc_dim = configs.phyc_dim
            ach_dim = configs.ach_dim
            sigle_dim = phyc_dim + ach_dim 
        if(self.frame_num == 2):
            pssm_dim = configs.pssm_dim
            psea_dim = configs.psea_dim
            sigle_dim = pssm_dim + psea_dim

        #卷积神经网络通道数目
        cnns_chanel = configs.cnn_chanel[self.frame_num]
        cnns_pools = configs.pools[self.frame_num]
        cnns_stride = configs.stride[self.frame_num]
        multiatten_dim = configs.atten_dim
        #局部特征经过卷积和注意力的算法
        cnns_out = math.floor(((2*window_size +1) - (cnns_pools-1)-1)/cnns_stride + 1)
        #input_dim = cnns_chanel*(multiatten_dim + cnns_out) + sigle_dim
        input_dim = cnns_chanel*(multiatten_dim + cnns_out)
        
        #局部特征输入的卷积网络层
        self.multi_CNN_L = nn.Sequential()
        self.multi_CNN_L.add_module("layer_convs",
                               ConvsLayerL())



        #深度神经网络的部分
        self.DNN1 = nn.Sequential()
        self.DNN1.add_module("DNN_layer1",
                            nn.Linear(input_dim,1024))
        self.DNN1.add_module("DNN_Norm1",nn.BatchNorm1d(1024))
        self.DNN1.add_module("ReLU1",
                            nn.ReLU())
        self.dropout_layer = nn.Dropout(self.dropout)
        

        self.DNN2 = nn.Sequential()
        self.DNN2.add_module("DNN_layer2",
                            nn.Linear(1024,256))
        self.DNN2.add_module("DNN_Norm2",nn.BatchNorm1d(256))
        self.DNN2.add_module("ReLU2",
                            nn.ReLU())
        

        self.outLayer = nn.Sequential(
            nn.Linear(256, self.class_num),
            nn.Sigmoid())


    def forward(self,single_features,local_features):
       
        #通过卷积层的局部特征
        features_l = local_features
        features_l = self.multi_CNN_L(features_l)

        #单独特征
        features_s = single_features

        #局部特征和单位点特征合成最终输入深度神经网络的特征
        #features = t.cat((features_s,features_l), 1)
        features = features_l

        features = self.DNN1(features)
        features =self.dropout_layer(features)
        features = self.DNN2(features)
        features =self.dropout_layer(features)
        features = self.outLayer(features)

        return features