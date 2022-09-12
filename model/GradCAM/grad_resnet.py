import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from model.GradCAM.gradcam import GradCAM,show_cam
from model.resnet50.resnet import resnet50,resnet34
from model.resnet50.ft import FeatureExtraction

def main():
    model=resnet50()
    weights_path="../../resources/output/resNet/re50.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    target_layers = [model.layer4[-1].conv3]
    eeg_datapath=["../../resources/output/3dm/eegmatrix3d_list05_00.npy"]
    fe=FeatureExtraction()
    eeg3dms=fe.split3DM(eeg_datapath,0,1,seconds=6)
    eeg3dms=eeg3dms.astype(np.float32)
    eeg3dms=torch.from_numpy(eeg3dms)
    for eeg3dm in eeg3dms:
        input_tensor = torch.unsqueeze(eeg3dm, dim=0)
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category=1
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam(grayscale_cam, use_rgb=True)
        plt.imshow(visualization)
        plt.show()
    pass

if __name__ == '__main__':
    main()

