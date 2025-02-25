import argparse
from collections import OrderedDict
from matplotlib import pyplot as plt
import torch
from CAM import GradCAM, CAM
from visualize import visualize
from modal import ConcatHNN1FC
from torch.utils.data.dataloader import DataLoader
import SimpleITK as sitk
from tool import resize_image,norm_image
import pandas as pd
import os
from torchvision.utils import save_image
from matplotlib import pyplot
import cv2 as cv
import cv2
import numpy as np


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



# image_path = 'C:/Users/medical/Desktop/seg/'




# def load_png(path, table_path):
#     d = 0
#     img_array = np.zeros([64, 64, 64])
#     fileList = os.listdir(path)
#     for file in fileList:
#         if file == '.DS_Store':
#             continue
#         fileName = path + file
#         png_data = cv.imread(fileName)
#         print(png_data.shape)
#         binary_data = cv.cvtColor(png_data, cv.COLOR_BGR2GRAY)
#         print(binary_data.shape)
#         # resize (64, 64)
#         resized_img = cv.resize(binary_data, (64, 64))
#         print(resized_img.shape)
#         img_array[:,:,d] = resized_img
#         d += 1
#     img_vol = torch.from_numpy(img_array)
#     img_vol = norm_image(img_vol)
#     clip = img_vol.unsqueeze(0).unsqueeze(0).float()
#
#
#     dir_name = os.path.dirname(os.path.dirname(os.path.dirname(table_path)))
#     txt_file_path = os.path.join(dir_name, 'tabular_vis.csv')
#
#     df = pd.read_csv(txt_file_path, header=None)
#     dataset = df.values
#     dataset = dataset.astype(float)
#     d = torch.from_numpy(dataset)
#
#     # [1,1,128,128,128]
#     return clip, d


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


    dir_name = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    txt_file_path = os.path.join(dir_name, 'tabular_vis.csv')

    df = pd.read_csv(txt_file_path, header=None)
    dataset = df.values
    dataset = dataset.astype(float)
    d = torch.from_numpy(dataset)


    return clip, d

image_path = 'E:/projects/pythonProject9/datasetADNI0923/train/AD//003_S_4373_bl'
tabular_path = 'tabular_vis.csv'






args = get_arguments()
model = ConcatHNN1FC()

state_dict = torch.load("best_model-fusion-0111.pt", map_location=lambda storage, loc: storage)


model.load_state_dict(state_dict)

target_layer = model.blockX.conv2

wrapped_model = CAM(model, target_layer)

model.eval()

with torch.no_grad():
    clip, tabular = load_image(image_path)
    # print(tabular)

    cam = wrapped_model(clip, tabular.to(torch.float32))
    # print(cam.shape)
    # print(clip.shape)
    heatmaps = visualize(clip, cam)
    # heatmaps = 255*heatmaps


# print(torch.min(heatmaps))
# print(torch.max(heatmaps))

save_path = 'E:/projects/pythonProject9/output10'
os.makedirs(save_path)
for i in range(clip.shape[2]):
    heatmap = heatmaps[:, :, i].squeeze()

    # print(heatmap.shape)

    save_image(heatmap, os.path.join(save_path, "{:0>3}.jpg".format(str(i))))
print("Done")








