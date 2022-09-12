import sys
import time

import  scipy.io as sio
import torch
from tqdm import tqdm
from model.resnet50.resnet import resnet50

from model.resnet50.ft import FeatureExtraction
import numpy as np
import os
import random
import torch.nn.functional as F


"""
测试数据读取
"""
# file_str='../../resources/data/wavedecdata/wave0'+str(1)+'.mat'
# temp=sio.loadmat(file_str)
# waveeg=temp['waveeg']
# cd1=waveeg[0][0]
# print(cd1.shape)
# cd2=waveeg[0][1]
# print(cd2.shape)
# cd3=waveeg[0][2]
# print(cd3.shape)
# cd4=waveeg[0][3]
# print(cd4.shape)
# cd5=waveeg[0][4]
# print(cd5.shape)
#========================================================================
# file_str='../../resources/data/originaldata/s0'+str(1)+'.mat'
# matrix_str='s0'+str(1)
# temp=sio.loadmat(file_str)[matrix_str][0]
# temp=temp[0]
# print(temp.shape)
# eegmatrix3d = np.zeros([7680,9,9])
# eegmatrix3d[:,0,3]+=temp[0,:]
# print('done')
#========================================================================
# fe=feature_extraction()
# fe.readsMat()
# fe.extractsMat()
#========================================================================
# label_fileName="../../resources/data/originaldata/arousal.mat"
# label=sio.loadmat(label_fileName)['arousal']
# label=label.flatten()
# print(label.shape)
#========================================================================
# eeg3DMs=[]
# loadName1="../../resources/output/eegmatrix3d_list20.npy"
# loadName2="../../resources/output/eegmatrix3d_list21.npy"
# eeg3DM1=np.load(loadName1)
# eeg3DM2=np.load(loadName2)
# eeg3DMs=np.concatenate((eeg3DM1,eeg3DM2),axis=0)
# print(eeg3DMs.shape)
#========================================================================
# a=['a','b','c','d','e','f']
# b=['1','1','0','2','3','1',]
# t_a=[]
# t_b=[]
# v_a=[]
# v_b=[]
# select_path=random.sample(a, k=int(len(a) * 0.5))
# for i in a:
#     if i in select_path:
#         v_a.append(i)
#         v_b.append(b[a.index(i)])
#     else:
#         t_a.append(i)
#         t_b.append(b[a.index(i)])
# print(t_a)
# print(t_b)
#====================for测试区=============================================
# root="../../resources/output/3dm/"
# namelist=[]
# for name in os.listdir(root):
#     namelist.append(os.path.join(root, name))
# namelist.sort()
# print(namelist)

# for i in range(1,16):
#     print(i)
# packNum=16
# loadBar=tqdm(range(packNum),file=sys.stdout)
# for i in loadBar:
#     time.sleep(2)
# start=0
# epochs=int(1280/16)
# for s in range(epochs):
#     print(start)
#     start+=16
#========================================================================
# eegdata=np.load("../../resources/output/3dm/eegmatrix3d_list01_00.npy")
# print(eegdata.shape)
# fe=FeatureExtraction()
# train_path,train_label,val_path,val_label=fe.readDataAndSplit(
#         root="../../resources/output/3dm/",val_rate=0.2
#     )
# data=fe.split3DM(train_path,0,16,6)
# label=fe.splitLabels(train_label,0,16,6)
# print(data.shape)
# print(label.shape)
#========================================================================
# tensor=[0., 0., 0., 0., 0., 1., 0., 1.]
# tensor=np.array(tensor).astype(np.int64)
# tensor = torch.as_tensor(tensor)
# print(F.one_hot(tensor))
#========================================================================
weights_path=["../../resources/output/resNet/re50.pth"]
model=resnet50()
model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
target_layers = [model.layer4[-1].bn3]
print(target_layers)