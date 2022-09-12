import sys

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.resnet50.ft import FeatureExtraction
from model.resnet50.resnet import resnet50,resnet34
import torch.optim as optim
import torch.nn.functional as F

class SmallerDataSet (data.Dataset):
    """
        60秒的片段裁剪成很多段，（6s）
    """
    def __init__(self,data,label):
        self.data=data.astype(np.float32)
        self.label=label.astype(np.int64)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, item):
        eeg3dm=self.data[item]
        eeg3dm=torch.from_numpy(eeg3dm)
        label=self.label[item]
        return eeg3dm,label

    @staticmethod
    def collate_fn(batch):
        eeg3dm, labels = tuple(zip(*batch))
        eeg3dm = torch.stack(eeg3dm, dim=0)
        # 如果可能，将数据转换为张量，共享数据并保留 autograd 历史记录
        labels = torch.as_tensor(labels)
        return eeg3dm, labels

# class SmallerDataSet (data.Dataset):
#     """
#         60秒的片段裁剪成很多段，（6s）
#     """
#     def __init__(self,data,label):
#         self.data=data.astype(np.float32)
#         self.label=label.astype(np.int64)
#     def __len__(self):
#         return len(self.label)
#     def __getitem__(self, item):
#         eeg3dm=self.data[item]
#         eeg3dm=torch.from_numpy(eeg3dm)
#         label=self.label[item]
#         return eeg3dm,label
#
#     @staticmethod
#     def collate_fn(batch):
#         eeg3dm, labels = tuple(zip(*batch))
#         eeg3dm = torch.stack(eeg3dm, dim=0)
#         # 如果可能，将数据转换为张量，共享数据并保留 autograd 历史记录
#         labels = torch.as_tensor(labels)
#         return eeg3dm, labels


class MyDataSet(data.Dataset):
    def __init__(self,path,label):
        self.path=path
        self.label=label
        fe=FeatureExtraction()
        for i in range(len(path)):
            self.eeg3dms=fe.split3DM(self.path,0,len(self.path),seconds=6)
            self.labels=fe.splitLabels(self.label,0,len(self.label),seconds=6)
    def __len__(self):
        return len(self.path)
    def __getitem__(self, item):
        eeg3dm=self.eeg3dms[item]
        eeg3dm=eeg3dm.astype(np.float32)
        eeg3dm=torch.from_numpy(eeg3dm)
        label=self.label[item].astype(np.int64)
        return eeg3dm,label

    @staticmethod
    def collate_fn(batch):
        eeg3dm, labels = tuple(zip(*batch))
        eeg3dm = torch.stack(eeg3dm, dim=0)
        # 如果可能，将数据转换为张量，共享数据并保留 autograd 历史记录
        labels = torch.as_tensor(labels)
        return eeg3dm, labels

def main():
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("using {} device.".format(device))
    fe=FeatureExtraction()
    train_path,train_label,val_path,val_label=fe.readDataAndSplit(
        root="../../resources/output/3dm/",val_rate=0.2
    )
    train_dataset = MyDataSet(path=train_path,label=train_label)
    val_dataset = MyDataSet(path=val_path,label=val_label)
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=8 ,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=val_dataset.collate_fn)
    loss_function = nn.CrossEntropyLoss()
    net = resnet50()

    net = resnet34()
    net.to(device)
    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=0.001, weight_decay=5E-2)

    epochs = 50
    best_acc = 0.0
    save_path = '../../resources/output/resNet/resNet50.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        sample_num = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            eeg3dms,labels=data
            logits = net(eeg3dms.to(device))
            loss = loss_function(logits, F.one_hot(labels).to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] " \
                             "loss:{:.3f}".format(epoch + 1,epochs,loss)
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for data in val_bar:
                eeg3dms,labels=data
                sample_num += eeg3dms.shape[0]
                outputs = net(eeg3dms.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)
            val_accurate = acc / sample_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
    print('Finished Training')

def trainWithSmallDM():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #epochs=20
    epochs = 20
    bestAcc = 0.0
    net = resnet50()
    #print(net.state_dict())
    net.to(device)
    pg = [p for p in net.parameters() if p.requires_grad]
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(pg, lr=0.001, weight_decay=5E-2)
    # 50个Epochs每一个Epochs都roll一次测试集和训练集
    fe = FeatureExtraction()
    train_path, train_label, val_path, val_label = fe.readDataAndSplit(
        root="../../resources/output/3dm/", val_rate=0.1
    )
    for epoch in range(epochs):
        startNum=0
        packNums=48
        seconds=6
        totalTrain=len(train_path)
        totalVal=len(val_path)
        sonEpochs = int(totalTrain / packNums)  # 子epochs   1024/16=72
        optimizer.zero_grad()
        for s in range(sonEpochs):
            eeg3dms=fe.split3DM(train_path,startNum,packNums,seconds)   #训练集分割，返回矩阵[int(60 / seconds)*packNum, spiltEeg, 9, 9]
            eeglabels=fe.splitLabels(train_label,startNum,packNums,seconds)     #标签集集扩展，返回矩阵[int(60 / seconds)*packNum,]
            startNum+=packNums
            trainDataset=SmallerDataSet(eeg3dms,eeglabels)
            trainLoader=DataLoader(dataset=trainDataset,batch_size=16,shuffle=True,pin_memory=True,collate_fn=trainDataset.collate_fn)
            runningLoss = 0.0
            trainBar = tqdm(trainLoader, file=sys.stdout)
            net.train()

            for step,data in enumerate(trainBar):
                eeg3dms,labels=data
                #optimizer.zero_grad()
                logits = net(eeg3dms.to(device))
                loss = lossFunction(logits, labels.to(device))
                #loss.backward()
                #optimizer.step()
                runningLoss += loss.item()
                trainBar.desc = "[epoch{} train] sonepoch[{}/{}] loss:{:.6f}".format(epoch + 1,s+1,sonEpochs,loss)
        loss.backward()
        optimizer.step()

        # validate
        #每一次都只选packNums个样本测试还是太少了，还是直接拿64个上吧
        eeg3dms = fe.split3DM(val_path, 0, totalVal,
                              seconds)  # 训练集分割，返回矩阵[int(60 / seconds)*packNum, spiltEeg, 9, 9]
        eeglabels = fe.splitLabels(val_label, 0, totalVal,
                                   seconds)  # 标签集集扩展，返回矩阵[int(60 / seconds)*packNum,]
        valDataset = SmallerDataSet(eeg3dms, eeglabels)
        valLoader = DataLoader(dataset=valDataset,
                               batch_size=16,
                               shuffle=False,
                               pin_memory=True,
                               collate_fn=valDataset.collate_fn)
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            valBar = tqdm(valLoader, file=sys.stdout)
            sampleNum = 0
            for data in valBar:
                eeg3dms, labels = data
                sampleNum+=labels.shape[0]
                outputs = net(eeg3dms.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc +=torch.eq(predict_y, labels.to(device)).sum().item()
                valBar.desc="valid epoch[{}/{}]".format(epoch + 1,epochs)
            valAccurate = acc / sampleNum
            if valAccurate > bestAcc:
                bestAcc = valAccurate
            print('[epoch %d] \'Accuracy\': %.6f ' % (epoch + 1, valAccurate))
        print('[epoch %d] \'bestAccuracy\': %.6f' % (epoch + 1, bestAcc))
    print('Finished Training')
    savePath="../../resources/output/resnet/re50"
    torch.save(net.state_dict(), savePath)

if __name__ == '__main__':
    #main()
    trainWithSmallDM()





