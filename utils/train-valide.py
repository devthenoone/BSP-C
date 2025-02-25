# E:\projects\pythonProject6

import os
from shutil import copy,rmtree
import random
import shutil

def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)



random.seed(0)
split_rate=0.2
cwd=os.getcwd()
data_root=os.path.join(cwd,"dataset")
# print(data_root)
# E:\projects\pythonProject6\dataset1
origin_flower_path=os.path.join(data_root,"train1")
# print(origin_flower_path)
# E:\projects\pythonProject6\dataset1\train1
assert os.path.exists(origin_flower_path)
flower_class=[cla for cla in os.listdir(origin_flower_path)
                  if os.path.isdir(os.path.join(origin_flower_path,cla))]
# print(flower_class)
# 生成了一个花类列表
# E:\projects\pythonProject6\dataset1\train
train_root=os.path.join(data_root,"train")
mk_file(train_root)
for cla in flower_class:
    mk_file(os.path.join(train_root,cla))

# E:\projects\pythonProject6\dataset1\val
val_root=os.path.join(data_root,"val")
mk_file(val_root)
for cla in flower_class:
    mk_file(os.path.join(val_root,cla))


for cla in flower_class:
    cla_path=os.path.join(origin_flower_path,cla)
    # print(cla_path)
    # 定位到原始数据集的各类花的文件夹
    images=os.listdir(cla_path)
    # print(images)
    num=len(images)
    # print(num)
    eval_index=random.sample(images,k=int(num*split_rate))
    k = int(num * split_rate)
    # print(k)

    for index,image in enumerate(images):
        if image in eval_index:
            image_path=os.path.join(cla_path,image)
            # print(image_path)
            new_path=os.path.join(val_root,cla)
            # print(new_path)
            shutil.move(image_path,new_path)
        else:
            image_path=os.path.join(cla_path,image)
            new_path=os.path.join(train_root,cla)
            shutil.move(image_path,new_path)
        print("\r[{}] processing [{}/{}]".format(cla,index+1,num),end="")
    print()

print("processing done!")
