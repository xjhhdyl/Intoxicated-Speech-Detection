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

if __name__ == '__main__':

    root = r"C:\Users\zrypz\PycharmProjects\Alcohol_detection_mix\data"
    trainVoice_path = ["\split_multisignal\\"]

    win_size = 0.025
    n_filters = 40

    tr_voicefile_list = traverse(root, trainVoice_path, search_fix=".wav")

    print("________________________________________________")
    print("Processing wav2logfbank...", flush=True)
    results = Parallel(n_jobs=2, backend="threading")(
        delayed(wav2logfbank)(i, win_size, n_filters) for i in tqdm(tr_voicefile_list)
    )

    # log-mel fbank 2 feature
    print("-------------------------------------------------")
    print("Preparing Training Dataset...", flush=True)

    tr_voicefile_list = traverse(root, trainVoice_path, search_fix=".fb" + str(n_filters))
    tr_text = traverse(root, trainVoice_path, return_label=True)
    tr_mmwavefile_list = traverse(root, trainsplit_ultrasound2wav_path, search_fix=".fb" + str(n_filters))
