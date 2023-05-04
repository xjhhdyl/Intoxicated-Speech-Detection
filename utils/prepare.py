import argparse
import os

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
# 2.上采样-下采样，相乘后的信号（去噪）
# 3.把超声波信号转换成wav格式，方便后面提取特征
# 4.利用相乘后的信号同时切割多模态信号---auditok
# 5.生成切割后的文件
# 6.划分训练集和测试集
# 7.提取特征
    args = parser.parse_args()
    main(args)

