#-*- encoding:utf8 -*-

import os
import time


import pickle
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal,xavier_normal_
from torch import nn
import torch.utils.data.sampler as sampler


from utils.config import DefaultConfig
from models.deep_ppi import DeepPPI
from generator import data_generator


from evaluation import compute_roc, compute_aupr, compute_mcc, micro_score,acc_score, compute_performance


configs = DefaultConfig()
THREADHOLD = 0.2

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        xavier_normal_(m.weight.data)
    elif isinstance(m,nn.Linear):
        xavier_normal_(m.weight.data)
    
def test(model, loader,path_dir,pre_num=1):

    # Model on eval mode
    model.eval()
    length = len(loader)
    result = []
    all_trues = []

    for batch_idx, (single_data, local_data, label) in enumerate(loader):
    
        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
                single_var = torch.autograd.Variable(single_data.cuda().float())
                local_var = torch.autograd.Variable(local_data.cuda().float())
            else:
                single_var = torch.autograd.Variable(single_data.float())
                local_var = torch.autograd.Variable(local_data.float())

        # Compute output
        output =  model(single_var,local_var)
        shapes = output.data.shape
        output = output.view(shapes[0]*shapes[1])
        result.append(output.data.cpu().numpy())
        all_trues.append(label.numpy())
        

    #caculate
    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(result, axis=0)

    auc = compute_roc(all_preds, all_trues)
    aupr = compute_aupr(all_preds, all_trues)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds,all_trues)
    acc = acc_score(predictions_max,all_trues)
    mcc = compute_mcc(predictions_max, all_trues)

    print(
        'acc:%0.6f,F_value:%0.6f, precision:%0.6f,recall:%0.6f,auc:%0.6f,aupr:%0.6f,mcc:%0.6f,threadhold:%0.6f\n' % (
        acc, f_max, p_max, r_max,auc, aupr,mcc,t_max))
    


    predict_result = {}
    predict_result["pred"] = all_preds
    predict_result["label"] = all_trues
    result_file = "{0}/test_pred_2_004_4.pkl".format(path_dir)
    with open(result_file,"wb") as fp:
        pickle.dump(predict_result,fp)


def predict(model_file,test_data,path_dir):
    test_sequences_file = ['data_cache/{0}_sequence_data.pkl'.format(key) for key in test_data]
    test_dssp_file = ['data_cache/{0}_dssp_data.pkl'.format(key) for key in test_data]
    test_pssm_file = ['data_cache/{0}_pssm_data.pkl'.format(key) for key in test_data]
    test_phyc_file = ['data_cache/{0}_phyc_data.pkl'.format(key) for key in test_data]
    test_ach_file = ['data_cache/{0}_ach_data.pkl'.format(key) for key in test_data]
    test_psea_file = ['data_cache/{0}_psea_data.pkl'.format(key) for key in test_data]
    test_label_file = ['data_cache/{0}_label.pkl'.format(key) for key in test_data]
    all_list_file = 'data_cache/all_dset_list.pkl'
    test_list_file = 'data_cache/test_list_4.pkl'
    
    # parameters
    batch_size = configs.batch_size

    print(test_list_file)

    # Datasets
    test_dataSet = data_generator.dataSet(test_sequences_file, test_pssm_file, test_dssp_file, test_phyc_file, test_ach_file, test_psea_file,
                                             test_label_file, all_list_file)
    
    # Models   
    with open(test_list_file,"rb") as fp:
        test_list = pickle.load(fp)

    #test_samples = sampler.SubsetRandomSampler(test_list)
    #切记千万不要随机测试集，否则就不是相同的顺序
    test_samples = test_list
    test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=batch_size,
                                              sampler=test_samples, pin_memory=(torch.cuda.is_available()),
                                               num_workers=5, drop_last=False)

    # Models
    model = DeepPPI()
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    test(model, test_loader,path_dir)

    print('Done!')



if __name__ == '__main__':

    path_dir = "./checkpoints/deep_ppi_saved_models"
    
    datas = ["dset186","dset164","dset72"]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    
    model_file = "{0}/DeepPPI_model.dat".format(path_dir)
    predict(model_file,datas,path_dir)

