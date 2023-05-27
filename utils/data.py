import sys
import torch
import numpy as np
import torchaudio as ta
from torch.utils.data import Dataset

# 二分类标签映射
label_map = {'sober': 0, 'intoxicate': 1}
listener_layers = 5

class mmWavoiceDataset(Dataset):
    def __init__(self, params, name="train"):
        self.params = params
        self.batch_size = params["data"]["batch_size"]
        if name == "test":
            self.batch_size = 2
        # the files voice and mmwave paths and id
        # idx voice_input mmwave_input label
        self.targets_dict = {}
        self.voicefile_list = []
        self.mmwavefile_list = []
        self.targets_real_target = {}
        with open(params["data"][name], "r", encoding="utf-8") as t:
            next(t)
            for line in t:
                parts = line.strip().split(",")
                sid = parts[0]
                voicepath = parts[1]
                mmwavepath = parts[2]
                label = label_map[parts[3]]
                self.targets_dict[sid] = label
                self.voicefile_list.append([sid, voicepath])
                self.mmwavefile_list.append([sid, mmwavepath])
        self.lengths = len(self.voicefile_list)

    def __getitem__(self, index):
        voice_utt_id, voice_path = self.voicefile_list[index]
        mmwave_utt_id, mmwave_path = self.mmwavefile_list[index]
        mmwave_feature = np.load(mmwave_path)
        voice_feature = np.load(voice_path)
        feature_length = max(voice_feature.shape[0], mmwave_feature.shape[0])
        targets = self.targets_dict[voice_utt_id]
        return voice_utt_id, voice_feature, mmwave_feature, feature_length, targets

    def __len__(self):
        return self.lengths


def collate_mmWavoice_fn(batch):
    # utt_id, voice_feature, mmwave_feature, feature_length, targets, targets_length
    utt_ids = [data[0] for data in batch]
    features_length = [data[3] for data in batch]
    max_feature_length = max(features_length)
    if max_feature_length % (2 ** listener_layers) != 0:
        max_feature_length += (2 ** listener_layers) - (max_feature_length % (2 ** listener_layers))
    padded_voicefeatures = []
    padded_mmwavefeatures = []
    padded_targets = []
    i = 0
    for utt_ids, voice_feat, mmwave_feat, feat_len, target in batch:
        sys.stdout.flush()
        padded_voicefeatures.append(
            np.pad(voice_feat, ((0, max_feature_length - feat_len), (0, 0)), mode="constant", constant_values=0.0, ))
        padded_mmwavefeatures.append(
            np.pad(mmwave_feat, ((0, max_feature_length - feat_len), (0, 0)), mode="constant", constant_values=0.0, ))
        padded_targets.append(target)
        i += 1
    voice_features = torch.FloatTensor(np.array(padded_voicefeatures))
    mmwave_features = torch.FloatTensor(np.array(padded_mmwavefeatures))
    features_length = torch.IntTensor(np.array(features_length))
    targets = torch.IntTensor(np.array(padded_targets))
    voice_features = {"voice_inputs": voice_features, "inputs_length": features_length}
    mmwave_features = {"mmwave_inputs": mmwave_features, "inputs_length": features_length}
    label = {"targets": targets}
    return utt_ids, voice_features, mmwave_features, label

class mmWavoiceLoader(object):
    def __init__(self, dataset, shuffle=False, ngpu=1, mode="ddp", num_workers=0):
        if ngpu > 1:
            if mode == "ddp":
                self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                self.sampler = None
        else:
            self.sampler = None
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size * ngpu,
            shuffle=shuffle,
            num_workers=num_workers * ngpu,
            pin_memory=True,
            sampler=self.sampler,
            collate_fn=collate_mmWavoice_fn,
            drop_last=True,
        )

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)