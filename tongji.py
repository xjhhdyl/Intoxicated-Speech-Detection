import librosa
import matplotlib.pyplot as plt
from pathlib import Path


def get_duration_mp3_and_wav(file_path):
    """
    获取mp3/wav音频文件时长
    :param file_path:
    :return:
    """
    duration = librosa.get_duration(filename=file_path)
    return duration

def plot_hist_of_meldata(datadirname):
    datadirp = Path(datadirname)

    timelens = []
    wavpaths = [x for x in datadirp.rglob('*.wav') if x.is_file()]

    for wavp in wavpaths:
        timelens.append(get_duration_mp3_and_wav(wavp))

    max_ = max(timelens)
    min_ = min(timelens)
    avg_ = int(sum(timelens) / len(timelens))
    print("max:{},min:{},avg:{}".format(max_, min_, avg_))

    plt.figure()
    plt.title("MelLens_hist_" + "max:{},min:{},avg:{}".format(max_, min_, avg_))
    plt.hist(timelens)
    plt.xlabel("mel length")
    plt.ylabel("numbers")
    plt.savefig("Mel_lengths_hist")
    plt.show()

if __name__ == "__main__":

    datadirname = r"C:\Users\zrypz\PycharmProjects\Alcohol_detection_mix\data\split_multisignal\sober"

    plot_hist_of_meldata(datadirname)