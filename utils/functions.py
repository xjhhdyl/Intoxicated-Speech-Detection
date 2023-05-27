import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import logfbank
from joblib import Parallel, delayed
from tqdm import tqdm

# 提取语音特征，作为网络的输入
def wav2logfbank(f_path, win_size, n_filters, nfft=512):
    (rate, sig) = wav.read(f_path)
    fbank_feat = logfbank(sig, rate, winlen=win_size, nfilt=n_filters, nfft=nfft)
    os.remove(f_path)
    np.save(f_path[:-3] + "fb" + str(n_filters), fbank_feat)


def traverse(root, path, search_fix=".flac"):
    f_list = []
    for p in path:
        p = root + p
        for sub_p in sorted(os.listdir(p)):
            for file in sorted(os.listdir(p + sub_p + "\\")):
                if search_fix in file:
                    file_path = p + sub_p + "\\" + file
                    f_list.append(file_path)
    return f_list
