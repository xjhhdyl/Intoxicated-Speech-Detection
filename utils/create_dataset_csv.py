import os
import pandas as pd
import re

"""
    生成数据集csv
"""

WAV_SOBER_DATA_DIR = "../data/voice/sober"  # 清醒语音数据的文件路径
WAV_INTOXICATE_DATA_DIR = "../data/voice/intoxicate"  # 醉酒语音数据的文件路径
DATASET = "../data/data.csv"  # 数据集存放的路径

# 0-清醒  1-醉酒
dataset_df = pd.DataFrame(columns=['wav_file_name', 'ultrasound_file_name', 'classID', 'class'])  # 创建数据集

wav_sober = os.listdir(WAV_SOBER_DATA_DIR)
wav_intoxicate = os.listdir(WAV_INTOXICATE_DATA_DIR)

for wav_file_name in wav_sober:
    number = re.findall("\d+", wav_file_name)
    ultrasound_file_name = "TEST_BINS"+number[0]+".csv"
    dataset_df.loc[len(dataset_df.index)] = [wav_file_name, ultrasound_file_name, 0, "sober"]  # 向数据集插入一条清醒数据记录

for wav_file_name in wav_intoxicate:
    number = re.findall("\d+", wav_file_name)
    ultrasound_file_name = "TEST_BINS"+number[0]+".csv"
    dataset_df.loc[len(dataset_df.index)] = [wav_file_name,ultrasound_file_name, 1,"intoxicate"]  # 向数据集插入一条醉酒数据记录

dataset_df.to_csv(DATASET, index=False)