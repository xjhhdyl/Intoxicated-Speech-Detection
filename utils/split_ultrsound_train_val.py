import os
from shutil import copy, rmtree
import random


def make_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


random.seed(0)

# 将数据集中10%的数据划分到验证集中
split_rate = 0.1
data_path = r'D:\chengxu\data\caodi'  # 数据集存放的地方，建议在程序所在的文件夹下新建一个data文件夹，将需要划分的数据集存放进去
data_root = r'D:\chengxu\data\cd'  # 这里是生成的训练集和验证集所处的位置，这里设置的是在当前文件夹下。

data_class = [cla for cla in os.listdir(data_path)]
print("数据的种类分别为：")
print(data_class)
# 建立保存训练集的文件夹
train_data_root = os.path.join(data_root, "train")  # 训练集的文件夹名称为 train
make_file(train_data_root)
for num_class in data_class:
    # 建立每个类别对应的文件夹
    make_file(os.path.join(train_data_root, num_class))

# 建立保存验证集的文件夹
val_data_root = os.path.join(data_root, "val")  # 验证集的文件夹名称为 val
make_file(val_data_root)
for num_class in data_class:
    # 建立每个类别对应的文件夹
    make_file(os.path.join(val_data_root, num_class))

for num_class in data_class:
    num_class_path = os.path.join(data_path, num_class)
    images = os.listdir(num_class_path)
    num = len(images)

    val_index = random.sample(images, k=int(num * split_rate))  # 随机抽取图片
    for index, image in enumerate(images):
        if image in val_index:

            data_image_path = os.path.join(num_class_path, image)
            val_new_path = os.path.join(val_data_root, num_class)
            copy(data_image_path, val_new_path)
        else:

            data_image_path = os.path.join(num_class_path, image)
            train_new_path = os.path.join(train_data_root, num_class)
            copy(data_image_path, train_new_path)
    print("\r[{}] split_rating [{}/{}]".format(num_class, index + 1, num), end="")  # processing bar
    print()

print("       ")
print("       ")
print("划分成功")