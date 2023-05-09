import os
import re
import torch
import torchaudio
import torchaudio.transforms as T
import argparse
import pandas as pd
import scipy.interpolate as spi
import soundfile
import numpy as np
from scipy import signal
import auditok

parser = argparse.ArgumentParser(description="preprocess.")

parser.add_argument(
    "--root",
    metavar="root",
    type=str,
    required=False,
    default="/data/alcohol_dataset/",
    help="Absolute file path to alcoholDataset.",
)

parser.add_argument(
    "--n_jobs",
    dest="n_jobs",
    action="store",
    default=-2,
    help="number of cpu available for preprocessing. \n -1: use all cpu, -2: use all  cpu but one",
)

parser.add_argument(
    "--n_filters",
    dest="n_filters",
    action="store",
    default=40,
    help="number of flters for fbank. (Default: 40)",
)
parser.add_argument(
    "--win_size",
    dest="win_size",
    action="store",
    default=0.025,
    help="Window size during feature extraction (Default : 0.025 [25ms])",
)
parser.add_argument(
    "--norm_x",
    dest="norm_x",
    action="store",
    default=False,
    help="Normalize features s.t. mean = 0 ,std=1",
)


def butter_lowpass(sample_rate, cut_off, order=5):
    """ 低通滤波器的设计 """
    nyq = 0.5 * sample_rate
    normal_cut_off = cut_off / nyq  # Wn 归一化的截止频率
    b, a = signal.butter(order, normal_cut_off, btype="low", analog=False)
    return b, a


def butter_lowpass_filtfilt(data, sample_rate, cut_off_frequency, order=5):
    """ 低通滤波器的执行，消除延迟 """
    b, a = butter_lowpass(sample_rate, cut_off_frequency, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def create_dataset_csv(WAV_SOBER_DATA_DIR, WAV_INTOXICATE_DATA_DIR, DATASET):
    # 0-清醒  1-醉酒
    dataset_df = pd.DataFrame(columns=['wav_file_name', 'ultrasound_file_name', 'classID', 'class'])  # 创建数据集

    wav_sober = os.listdir(WAV_SOBER_DATA_DIR)
    wav_intoxicate = os.listdir(WAV_INTOXICATE_DATA_DIR)

    for wav_file_name in wav_sober:
        number = re.findall("\d+", wav_file_name)  # 提取出字符串中的编号
        ultrasound_file_name = "TEST_BINS" + number[0] + ".csv"
        ULTR_SOBER_DIR = WAV_SOBER_DATA_DIR.replace("voice", "ultrasound")
        dataset_df.loc[len(dataset_df.index)] = [WAV_SOBER_DATA_DIR + "/" + wav_file_name,
                                                 ULTR_SOBER_DIR + "/" + ultrasound_file_name, 0,
                                                 'sober']  # 向数据集插入一条清醒数据记录

    for wav_file_name in wav_intoxicate:
        number = re.findall("\d+", wav_file_name)
        ultrasound_file_name = "TEST_BINS" + number[0] + ".csv"
        ULTR_INTOXICATE_DIR = WAV_INTOXICATE_DATA_DIR.replace("voice", "ultrasound")
        dataset_df.loc[len(dataset_df.index)] = [WAV_INTOXICATE_DATA_DIR + "/" + wav_file_name,
                                                 ULTR_INTOXICATE_DIR + "/" + ultrasound_file_name, 1,
                                                 'intoxicate']  # 向数据集插入一条醉酒数据记录
    dataset_df.to_csv(DATASET, index=False)


def create_split_dataset_csv(mix_data_dir, split_dataset):
    # 0-清醒  1-醉酒
    split_dataset_df = pd.DataFrame(columns=['mix_file_path', 'ultrasound2wav_file_path', 'classID', 'class'])  # 创建数据集
    mix_sober_data_dir = mix_data_dir + "/sober"
    mix_intoxicate_data_dir = mix_data_dir + "/intoxicate"

    mix_sober = os.listdir(mix_sober_data_dir)
    mix_intoxicate = os.listdir(mix_intoxicate_data_dir)

    for mix_sober_file_name in mix_sober:
        number = re.findall("\d+", mix_sober_file_name)  # 提取出字符串中的编号
        ultrasound2wav_file_name = "TEST_BINS" + number[0] + ".wav"
        u2w_sober_dir = mix_sober_data_dir.replace("multilsignal", "ultrasound2wav")
        split_dataset_df.loc[len(split_dataset_df.index)] = [mix_sober_data_dir + "/" + mix_sober_file_name,
                                                             u2w_sober_dir + "/" + ultrasound2wav_file_name, 0,
                                                             'sober']  # 向数据集插入一条清醒数据记录

    for mix_intoxicate_file_name in mix_intoxicate:
        number = re.findall("\d+", mix_intoxicate_file_name)  # 提取出字符串中的编号
        ultrasound2wav_file_name = "TEST_BINS" + number[0] + ".wav"
        u2w_intoxicate_dir = mix_intoxicate_data_dir.replace("multilsignal", "ultrasound2wav")
        split_dataset_df.loc[len(split_dataset_df.index)] = [mix_intoxicate_data_dir + "/" + mix_intoxicate_file_name,
                                                             u2w_intoxicate_dir + "/" + ultrasound2wav_file_name, 1,
                                                             'intoxicate']  # 向数据集插入一条清醒数据记录
    split_dataset_df.to_csv(split_dataset, index=False)


def upAnddown(wav_file_path, bins_file_path):
    waveform, sample_rate = torchaudio.load(wav_file_path)  # 读取音频文件

    esd_df = pd.read_csv(bins_file_path)  # 读取ESD数据文件
    time_df = esd_df.iloc[:, 0]  # 获取时间戳的那一列
    bins_df = esd_df.iloc[:, 1:-1]  # 获取bins的列
    max = bins_df.max(axis=1)  # 找到bins中每一行的最大值，并且赋值给max列
    esd_time = (time_df - time_df[0]) / 1000  # bins的时间序列

    # 音频下采样至16Khz
    resample_rate = 16000
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)

    # 超声波上采样至16Khz
    num_channels, num_frames = resampled_waveform.shape
    time = torch.arange(0, num_frames) / resample_rate  # 音频的时间点
    ipo1 = spi.splrep(esd_time.values, max.values, k=1)  # 样本点导入，生成参数
    upsample_esd = spi.splev(time, ipo1)  # 根据观测点和样条参数，生成插值，观测点设置为音频的时间坐标

    # 超声波文件转化为wav文件
    ultrasound2wav_path = bins_file_path.replace('ultrasound', 'ultrasound2wav').replace('csv', 'wav')
    soundfile.write(ultrasound2wav_path, upsample_esd, resample_rate)

    # 超声波和音频信号相乘
    multilsignal = np.multiply(upsample_esd, resampled_waveform.numpy().reshape(-1, 1).squeeze())
    multilsignal_path = bins_file_path.replace('ultrasound', 'multilsignal').replace('TEST_BINS', 'mix').replace('csv',
                                                                                                                 'wav')
    butter_lowpass_multilsignal = butter_lowpass_filtfilt(multilsignal, resample_rate, 1000)  # 低通滤波，截至频率为1000Hz
    soundfile.write(multilsignal_path, butter_lowpass_multilsignal, resample_rate)


def split_signal(mix_data_dir, split_dataset):
    create_split_dataset_csv(mix_data_dir, split_dataset)  # 创建分割数据集

    split_mix_dir = 'data/split_multisignal/'

    split_data_df = pd.read_csv(split_dataset)  # 读取数据集的文件
    for row in split_data_df.itertuples():  # 按行遍历
        mix_file_path = getattr(row, 'mix_file_path')
        signal_class = mix_file_path.split('/')[2]
        audio_regions = auditok.split(
            mix_file_path,  # 通过getattr(row, ‘name')获取元素
            min_dur=0.2,  # minimum duration of a valid audio event in seconds
            max_dur=4,  # maximum duration of an event
            max_silence=0.3,  # maximum duration of tolerated continuous silence within an event
            energy_threshold=55  # threshold of detection
        )

        for i, r in enumerate(audio_regions):
            # Regions returned by `split` have 'start' and 'end' metadata fields
            print("Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))
            print(r.meta.end)
            # region's metadata can also be used with the `save` method
            # (no need to explicitly specify region's object and `format` arguments)

            file = mix_file_path.split('/')[3]
            split_mix_file_name = os.path.splitext(file)[0]  # mix1
            split_mix_file_path = split_mix_dir + signal_class + "/" + split_mix_file_name + "_" + str(i) + ".wav"
            r.save(split_mix_file_path)


def main(args):
    root = args.root
    target_path = root + "/processed/"
    trainVoice_path = ["train_voice/"]
    trainultrasound_path = ["train_ultrasound/"]
    devVoice_path = ["test_voice/"]
    devultrasound_path = ["test_ultrasound/"]
    n_jobs = args.n_jobs
    n_filters = args.n_filters
    win_size = args.win_size
    norm_x = args.norm_x

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    print("-------------Processing Datasets--------------")
    print("Training Voice sets :", trainVoice_path)
    print("Training ultrasound sets", trainultrasound_path)
    print("Validation Voice sets:", devVoice_path)
    print("Validation ultrasound sets", devultrasound_path)
    print("-----------------------------------------------")


if __name__ == '__main__':
    # 1.从原始数据集导出描述数据集的csv
    # 2.音频下采样 -> 超声波上采样 -> 超声波信号和音频信号相乘
    # 3.把超声波信号转换成wav格式，相乘信号低通滤波
    # 4.利用auditok同时切割多模态信号
    # 5.生成切割后的文件
    # 6.划分训练集和测试集
    # 7.提取Log-mel Filterbank Coefficients特征

    # WAV_SOBER_DATA_DIR = "data/voice/sober"  # 清醒语音数据的文件路径
    # WAV_INTOXICATE_DATA_DIR = "data/voice/intoxicate"  # 醉酒语音数据的文件路径
    # DATASET = "data/data.csv"  # 数据集存放的路径
    #
    # create_dataset_csv(WAV_SOBER_DATA_DIR, WAV_INTOXICATE_DATA_DIR, DATASET)  # 生成原始数据集的描述文件csv
    # data_df = pd.read_csv(DATASET)  # 读取数据集的文件
    # for row in data_df.itertuples():  # 按行遍历
    #     # 通过getattr(row, ‘name')获取元素
    #     upAnddown(getattr(row, 'wav_file_name'), getattr(row, 'ultrasound_file_name'))  # 对音频上采样，超声波下采样

    split_signal("data/multilsignal", "data/split.csv")

    # args = parser.parse_args()
    # main(args)
