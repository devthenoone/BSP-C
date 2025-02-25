import argparse
from collections import OrderedDict
from matplotlib import pyplot as plt
import torch
from CAMimage import GradCAM, CAM
from visualizeimage import visualize
from modal import HeterogeneousResNet
from torch.utils.data.dataloader import DataLoader
import SimpleITK as sitk
from tool import resize_image,norm_image
import pandas as pd
import os
from torchvision.utils import save_image
from matplotlib import pyplot
import cv2 as cv
import cv2


def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(description="visualization")
    parser.add_argument(
        "--video_dir", type=str, default="./videos", help="path of a config file"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./cams", help="path of a config file"
    )

    return parser.parse_args()

def load_image(path):
    series_reader = sitk.ImageSeriesReader()
    fileNames = series_reader.GetGDCMSeriesFileNames(path)
    series_reader.SetFileNames(fileNames)
    images = series_reader.Execute()


    images = resize_image(images, (64, 64, 64), resamplemethod=sitk.sitkLinear)
    img_array = sitk.GetArrayFromImage(images)
    img_vol = torch.from_numpy(img_array)
    img_vol = norm_image(img_vol)
    clip = img_vol.unsqueeze(0).unsqueeze(0).float()


    return clip

image_path = 'E:/projects/pythonProject9/datasetADNI0923/train/AD//003_S_4373_bl'



args = get_arguments()
model = HeterogeneousResNet()

state_dict = torch.load("best_model1image-only-zhuanhuan.pt", map_location=lambda storage, loc: storage)


model.load_state_dict(state_dict)

target_layer = model.block4.conv2

wrapped_model = CAM(model, target_layer)

model.eval()

with torch.no_grad():
    clip = load_image(image_path)

    cam = wrapped_model(clip)
    # print(cam.shape)
    # print(clip.shape)
    heatmaps = visualize(clip, cam)
    # heatmaps = 255*heatmaps


# print(torch.min(heatmaps))
# print(torch.max(heatmaps))

save_path = 'E:/projects/pythonProject9/output'
os.makedirs(save_path)
for i in range(clip.shape[2]):
    heatmap = heatmaps[:, :, i].squeeze()

    # print(heatmap.shape)

    save_image(heatmap, os.path.join(save_path, "{:0>3}.jpg".format(str(i))))
print("Done")








