import SimpleITK as sitk
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import glob
import cv2
import numpy as np
from skimage import transform
import os
import pandas as pd
import torchvision
import csv
from tool import resize_image,norm_image


class ADMdataset(Dataset):
    def __init__(self, data_txt):
        self.data_txt=data_txt
        self.datasets=[ ]

        for file in open(self.data_txt,'r'):
            image_file=file.strip('\n').split(' ')[0]
            image_label=file.strip('\n').split(' ')[1]
            self.datasets.append([image_file, image_label])
            # print(self.datasets)

    def __getitem__(self, idx):
        image = self.datasets[idx][0]
        dir_name = os.path.dirname(os.path.dirname(os.path.dirname(self.datasets[idx][0])))
        txt_file_path = os.path.join(dir_name, 'tabular.csv')
        series_reader = sitk.ImageSeriesReader()
        fileNames = series_reader.GetGDCMSeriesFileNames(self.datasets[idx][0])
        series_reader.SetFileNames(fileNames)
        images = series_reader.Execute()
        images = resize_image(images, (64, 64, 64), resamplemethod=sitk.sitkLinear)
        img_array = sitk.GetArrayFromImage(images)
        img_vol = torch.from_numpy(img_array)
        img_vol = norm_image(img_vol)
        img_vol = img_vol.unsqueeze(0).float()
        image_label = self.datasets[idx][1]
        image_label = float(image_label)
        c = os.path.basename(image)
        list = []
        with open(txt_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['PTID'] == c:
                    del row['PTID']
                    for v in row.values():
                        v = float(v)
                        list.append(v)
                    data = torch.tensor(list)
        image_label = torch.tensor(image_label)
        return img_vol, image_label, data
    def __len__(self):
        return len(self.datasets)


