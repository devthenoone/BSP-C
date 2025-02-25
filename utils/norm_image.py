import torch


def norm_image(image):
    max = torch.max(image)
    min = torch.min(image)
    image = (image - min) / (max - min)
    return image