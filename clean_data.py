import os

def del_files(dir_path):
    if os.path.isfile(dir_path):
        try:
            os.remove(dir_path) # 这个可以删除单个文件，不能删除文件夹
        except BaseException as e:
            print(e)
    elif os.path.isdir(dir_path):
        file_lis = os.listdir(dir_path)
        for file_name in file_lis:
            # if file_name != 'wibot.log':
            tf = os.path.join(dir_path, file_name)
            del_files(tf)

if __name__ == "__main__":

    datadirname = [r"C:\Users\zrypz\PycharmProjects\Alcohol_detection_mix\data\multilsignal",
                   r"C:\Users\zrypz\PycharmProjects\Alcohol_detection_mix\data\multisignal_dataset",
                   r"C:\Users\zrypz\PycharmProjects\Alcohol_detection_mix\data\processed",
                   r"C:\Users\zrypz\PycharmProjects\Alcohol_detection_mix\data\split_multisignal",
                   r"C:\Users\zrypz\PycharmProjects\Alcohol_detection_mix\data\split_u2w",
                   r"C:\Users\zrypz\PycharmProjects\Alcohol_detection_mix\data\u2w_dataset",
                   r"C:\Users\zrypz\PycharmProjects\Alcohol_detection_mix\data\ultrasound2wav"]

    for pathname in datadirname:
        del_files(pathname)
        print(f"delete{pathname}")
