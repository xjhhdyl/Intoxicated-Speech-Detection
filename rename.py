import os
import re

path = r"D:\alcohol_dataset\rename"
# 批量重命名文件
for filename in os.listdir(path):
    number = re.findall("\d+", filename)  # 提取出字符串中的编号
    num = int(number[0]) - 2
    newname = "getWav" + str(num) + ".wav"
    os.rename(path + "\\getWav" + number[0] + ".wav", path + "\\" + newname)
