#-*- encoding:utf8 -*-

import os
import time
import pickle
import torch as t
import numpy as np
from torch.utils import data

#my lib
from utils.config import DefaultConfig 


class dataSet(data.Dataset):
    def __init__(self,sequences_file=None,pssm_file=None, dssp_file=None, phyc_file=None, ach_file=None, psea_file=None, 
                 label_file=None, protein_list_file=None):
        super(dataSet,self).__init__()
        
        self.Config = DefaultConfig()
        #重点关注这块的数据传递
        self.frame_num = self.Config.frame_num
        self.window_size = self.Config.windows_size[self.frame_num]

        #以下根据框架编号，初始化不同的参数
        #其实有些文件是用不到的，暂时保留了文件全部传递的设置
        #框架编号为0的时候，使用seq和dssp特征
        if(self.frame_num == 0):
            self.all_sequences = []
            for seq_file in sequences_file:
                with open(seq_file,"rb") as fp_seq:
                    temp_seq  = pickle.load(fp_seq)
                self.all_sequences.extend(temp_seq)
            
            self.all_dssp = []
            for dp_file in dssp_file: 
                with open(dp_file,"rb") as fp_dssp:
                    temp_dssp  = pickle.load(fp_dssp)
                self.all_dssp.extend(temp_dssp)
        
        #框架编号为1的时候，使用phyc和ach特征
        if(self.frame_num == 1):
            self.all_phyc = []
            for ph_file in phyc_file: 
                with open(ph_file,"rb") as fp_phyc:
                    temp_phyc  = pickle.load(fp_phyc)
                self.all_phyc.extend(temp_phyc)

            self.all_ach = []
            for ac_file in ach_file: 
                with open(ac_file,"rb") as fp_ach:
                    temp_ach  = pickle.load(fp_ach)
                self.all_ach.extend(temp_ach)
        
        #框架编号为2的时候，使用pssm和psea特征
        if(self.frame_num == 2):
            self.all_pssm = []
            for pm_file in pssm_file: 
                with open(pm_file,"rb") as fp_pssm:
                    temp_pssm = pickle.load(fp_pssm)
                self.all_pssm.extend(temp_pssm)

            self.all_psea = []
            for ps_file in psea_file: 
                with open(ps_file,"rb") as fp_psea:
                    temp_psea  = pickle.load(fp_psea)
                self.all_psea.extend(temp_psea)
        
        #从代码的完整性上来说，应该在这里加一段frame_num越界的提示


        #蛋白质位点是否具有相互作用的标志信息
        self.all_label = []
        for lab_file in label_file: 
            with open(lab_file, "rb") as fp_label:
                temp_label = pickle.load(fp_label)
            self.all_label.extend(temp_label)

        with open(protein_list_file, "rb") as list_label:
            self.protein_list = pickle.load(list_label)

         

        
    def __getitem__(self,index):
        
        count,id_idx,ii,dset,protein_id,seq_length = self.protein_list[index]
        #局部特征的作用区间，决定特征起止范围
        window_size = self.window_size
        id_idx = int(id_idx)
        win_start = ii - window_size
        win_end = ii + window_size
        seq_length = int(seq_length)
        label_idx = (win_start+win_end)//2 #这个其实不就是ii吗，不明白为什么要这样写，暂时保留
        
        #载入框架编号
        frame_num = self.frame_num

        #单点特征组合方式
        single_features = []
        if(frame_num == 0):
            #seq
            idx = self.all_sequences[id_idx][ii]
            acid_one_hot = [0 for i in range(20)]
            acid_one_hot[idx] = 1
            single_features.extend(acid_one_hot)
            #dssp
            try:
                dssp_val = self.all_dssp[id_idx][ii]
            except:
                dssp_val = [0 for i in range(9)]
            single_features.extend(dssp_val)
        if(frame_num == 1):
            #phyc
            phyc_val = self.all_phyc[id_idx][ii]
            single_features.extend(phyc_val)
            #ach
            ach_val = self.all_ach[id_idx][ii]
            single_features.extend(ach_val)
        if(frame_num == 2):
            #pssm
            pssm_val = self.all_pssm[id_idx][ii]
            single_features.extend(pssm_val)
            #psea
            psea_val = self.all_psea[id_idx][ii]
            single_features.extend(psea_val)


        #局部特征组合方式
        local_features = []
        #这里其实还应该有更一般的写法，就是所有零变量的维度，应该由config.py中读出，而不是直接写死，
        #用来确保变量维度的修改一致性，现在为了代码更好理解，暂时保留简单写法
        while win_start<0:
            data = []
            if(frame_num == 0):
                #seq
                acid_one_hot = [0 for i in range(20)]
                data.extend(acid_one_hot)
                #dssp
                dssp_zero_vector = [0 for i in range(9)]
                data.extend(dssp_zero_vector)

            if(frame_num == 1):
                #phyc
                phyc_zero_vector = [0 for i in range(10)]
                data.extend(phyc_zero_vector)
                #ach
                ach_zero_vector = [0 for i in range(2)]
                data.extend(ach_zero_vector)

            if(frame_num == 2):
                #pssm
                pssm_zero_vector = [0 for i in range(20)]
                data.extend(pssm_zero_vector)
                #psea
                psea_zero_vector = [0 for i in range(30)]
                data.extend(psea_zero_vector)

            local_features.append(data)
            win_start += 1
       
        valid_end = min(win_end,seq_length-1)
        while win_start<=valid_end:
            data = []
            if(frame_num == 0):
                #seq
                idx = self.all_sequences[id_idx][win_start]
                acid_one_hot = [0 for i in range(20)]
                acid_one_hot[idx] = 1
                data.extend(acid_one_hot)
                #dssp
                try:
                    dssp_val = self.all_dssp[id_idx][win_start]
                except:
                    dssp_val = [0 for i in range(9)]
                data.extend(dssp_val)

            if(frame_num == 1):
                #phyc
                phyc_val = self.all_phyc[id_idx][win_start]
                data.extend(phyc_val)
                #ach
                ach_val = self.all_ach[id_idx][win_start]
                data.extend(ach_val)

            if(frame_num == 2):
                #pssm
                pssm_val = self.all_pssm[id_idx][win_start]
                data.extend(pssm_val)
                #psea
                psea_val = self.all_psea[id_idx][win_start]
                data.extend(psea_val)

            local_features.append(data)
            win_start += 1

        while win_start<=win_end:
            data = []
            if(frame_num == 0):
                #seq
                acid_one_hot = [0 for i in range(20)]
                data.extend(acid_one_hot)
                #dssp
                dssp_zero_vector = [0 for i in range(9)]
                data.extend(dssp_zero_vector)

            if(frame_num == 1):
                #phyc
                phyc_zero_vector = [0 for i in range(10)]
                data.extend(phyc_zero_vector)
                #ach
                ach_zero_vector = [0 for i in range(2)]
                data.extend(ach_zero_vector)

            if(frame_num == 2):
                #pssm
                pssm_zero_vector = [0 for i in range(20)]
                data.extend(pssm_zero_vector)
                #psea
                psea_zero_vector = [0 for i in range(30)]
                data.extend(psea_zero_vector)

            local_features.append(data)
            win_start += 1


        #蛋白质位点是否相互作用的标志
        label = self.all_label[id_idx][label_idx]
        label = np.array(label,dtype=np.float32)

        #对局部特征数据处理
        local_features = np.stack(local_features)
        local_features = local_features[np.newaxis,:,:]
        #单点特征数据处理
        single_features = np.stack(single_features)


        return single_features,local_features,label
                

    def __len__(self):
    
        return len(self.protein_list)