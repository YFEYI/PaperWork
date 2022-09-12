import sys

import scipy.io as sio
import numpy as np
import random
import os

from tqdm import tqdm


class FeatureExtraction:

    def readsMat(self):
        """
            读取原始的数据s01-s32.mat
        :return:[32,40,1]
        """
        data=[]
        for i in range(1,33):
            if i<10:
                file_str='../../resources/data/originaldata/s0'+str(i)+'.mat'
                matrix_str = 's0' + str(i)
            else:
                file_str = '../../resources/data/originaldata/s' + str(i) + '.mat'
                matrix_str = 's' + str(i)
            eeg=sio.loadmat(file_str)[matrix_str]
            data.append(eeg)
        self.extracted_data=np.array(data)

    def extractsMat(self):
        """
            将原始数据转化成3D矩阵，一[32,7680]->[7680,9,9]
            电脑太菜了不能把整个1280个3D矩阵保存成一个.npy文件，所以又存了1280个55555555.
            整整5个G！！
            我绷不住了，这个python的sort()方法！保存的文件名要在一位数的前面加个0
        :return:
        """
        num = 0
        for i in range(32):
            data = self.extracted_data
            data=data[i]    #(40,1)

            for j in range(40):
                temp=data[j][0]    #这里突然变成了1维ndarray，在拆一次(32,7680)
                eegmatrix3d = np.zeros([7680,9,9])
                print(f'正在创建第{i+1}个人的第{j+1}条数据（{num}/1280）')
                name = "../../resources/output/3dm/eegmatrix3d_list"
                train_data_path=self.makeSavefileName(name,i,j)
                eegmatrix3d=self.make3DMatrix(channels=32,eegdata=temp,eegmatrix3d=eegmatrix3d)
                np.save(train_data_path, eegmatrix3d)
                num+=1

    def readDataAndSplit(self,root:str,val_rate=0.2):
        """
            将1280个文件组成list，按比例分为训练集和测试集
        :param root:文件存放的根路径
        :param val_rate:测试集所占比例
        :return:
        """
        random.seed(22)
        assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
        pathlist = []
        for name in os.listdir(root):
            # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表 list[](只有文件名)
            # os.path.join 	把目录和文件名合成一个路径
            path=os.path.join(root, name)
            pathlist.append(path)
        pathlist.sort()
        labellist=self.extractArousalMat()
        train_path = []  # 存储训练集
        train_label = []  # 存储训练集索引信息
        val_path = []  # 存储验证集
        val_label = []  # 存储验证集索引信息
        select_path=random.sample(pathlist, k=int(len(pathlist) * val_rate))
        for path in pathlist:
            if path in select_path:
                val_path.append(path)
                val_label.append(labellist[pathlist.index(path)])
            else:
                train_path.append(path)
                train_label.append(labellist[pathlist.index(path)])
        print("{} for training.".format(len(train_path)))
        print("{} for validation.".format(len(val_path)))
        return train_path,train_label,val_path,val_label



    def make3DMatrix(self,channels:int,eegdata,eegmatrix3d):
        """
            [32,7680]->[7680,9,9]
        :param channels: 32通道
        :return: eegmatrix3d
        """
        if channels==32:
            eegmatrix3d[:, 0, 3] += eegdata[0, :]
            eegmatrix3d[:, 1, 3] += eegdata[1, :]
            eegmatrix3d[:, 2, 0] += eegdata[2, :]
            eegmatrix3d[:, 2, 2] += eegdata[3, :]
            eegmatrix3d[:, 3, 3] += eegdata[4, :]
            eegmatrix3d[:, 3, 1] += eegdata[5, :]
            eegmatrix3d[:, 4, 0] += eegdata[6, :]
            eegmatrix3d[:, 4, 2] += eegdata[7, :]
            eegmatrix3d[:, 5, 3] += eegdata[8, :]
            eegmatrix3d[:, 5, 1] += eegdata[9, :]
            eegmatrix3d[:, 6, 0] += eegdata[10, :]
            eegmatrix3d[:, 6, 2] += eegdata[11, :]
            eegmatrix3d[:, 6, 4] += eegdata[12, :]
            eegmatrix3d[:, 7, 3] += eegdata[13, :]
            eegmatrix3d[:, 8, 3] += eegdata[14, :]
            eegmatrix3d[:, 8, 4] += eegdata[15, :]
            eegmatrix3d[:, 8, 5] += eegdata[16, :]
            eegmatrix3d[:, 7, 5] += eegdata[17, :]
            eegmatrix3d[:, 6, 6] += eegdata[18, :]
            eegmatrix3d[:, 6, 8] += eegdata[19, :]
            eegmatrix3d[:, 5, 7] += eegdata[20, :]
            eegmatrix3d[:, 5, 5] += eegdata[21, :]
            eegmatrix3d[:, 4, 6] += eegdata[22, :]
            eegmatrix3d[:, 4, 8] += eegdata[23, :]
            eegmatrix3d[:, 3, 7] += eegdata[24, :]
            eegmatrix3d[:, 3, 5] += eegdata[25, :]
            eegmatrix3d[:, 2, 6] += eegdata[26, :]
            eegmatrix3d[:, 2, 8] += eegdata[27, :]
            eegmatrix3d[:, 1, 5] += eegdata[28, :]
            eegmatrix3d[:, 0, 5] += eegdata[29, :]
            eegmatrix3d[:, 2, 4] += eegdata[30, :]
            eegmatrix3d[:, 4, 4] += eegdata[31, :]
            return eegmatrix3d
        else:
            # 未实现62电极预留
            return None

    def makeSavefileName(self,root,i,j):
        """
            别问我为什么两个if写一个方法，问sort
        :param root: 保存的文件夹目录
        :param i:
        :param j:
        :return: 生成的文件名
        """
        if i < 10:
            fileName = root + "0" + str(i)
        else:
            fileName = root + str(i + 1)
        if j < 10:
            fileName = fileName + '_0' + str(j)
        else:
            fileName = fileName + '_' + str(j)
        return fileName

    def readWaveMat(self):
        """
            读取小波变换之后的数据 (有问题，重新提取数据)
        :return: waveeg[1,5] 分别存放着Gamma，Beta，Alpha，Theta，Delta
        """
        for i in range(1,33):
            if i<10:
                file_str='../../resources/data/wavedecdata/wave0'+str(i)+'.mat'
            else:
                file_str = '../../resources/data/wavedecdata/wave' + str(i) + '.mat'
            waveeg=sio.loadmat(file_str)['waveeg']
        return waveeg

    def extractArousalMat(self):
        """
            读取唤醒mat文件
        :return: ndarray（1280,）
        """
        label_fileName = "../../resources/data/originaldata/arousal.mat"
        label = sio.loadmat(label_fileName)['arousal']
        label = label.flatten()
        return label

    def split3DM(self,paths, i, packNum=16, seconds=6):
        """
            分割60秒的3d矩阵
        :param paths: 所有的文件路径
        :param i: 从i个路径开始
        :param packNum: 一次训练会载入多少个文件
        :param seconds: 分割的精度
        :return:
        """
        # 加载开始的文件
        eeg3dm = np.load(paths[i])
        spiltNum = int(60 / seconds)  # 分成几分
        spiltEeg = int(7680 / spiltNum)  # 一份多长
        eeg3dms = eeg3dm.reshape(spiltNum, spiltEeg, 9, 9)
        loadBar = tqdm(range(1, packNum), file=sys.stdout)
        if packNum>1:
            for n in loadBar:
            #for n in range(1, packNum):
                eeg3dm = np.load(paths[i + n])
                temp = eeg3dm.reshape(spiltNum, spiltEeg, 9, 9)
                eeg3dms = np.concatenate([eeg3dms, temp], axis=0)
                loadBar.desc = "加载训练集".format(seconds,packNum)
        return eeg3dms

    def splitLabels(self,labels, i,packNum=16, seconds=6):
        """
            分割label使一一对应
        :param labels: 【1280,】
        :param i: 开始标记
        :param packNum: 打包个数
        :return: 分割后的label
        """
        copyNum = int(60 / seconds)  # 分成几分
        newLabels=np.zeros(packNum*copyNum)     #新的label列表
        num=0
        #loadBar = tqdm(range(packNum), file=sys.stdout)
        #for n in loadBar:
        for n in range(packNum):
            temp=labels[i+n]
            for c in range(copyNum):
                newLabels[num]=temp
                num+=1
            #loadBar.desc = "扩展标签集".format(seconds, packNum)
        return newLabels

if __name__ == '__main__':
    fe = FeatureExtraction()
    if os.path.exists("../../resources/output/3dm/eegmatrix3d_list00_00.npy"):
        train_path,train_label,val_path,val_label=fe.readDataAndSplit(root="../../resources/output/3dm/",val_rate=0.2)
        print("done")
    else:
        fe.readsMat()
        fe.extractsMat()