from torch.utils.data import DataLoader, Dataset
import random
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import torch.nn as nn
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from transformers import Wav2Vec2Processor
import math

def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1. - alpha) * xb
    return torch.log(x + torch.finfo(x.dtype).eps)

class Mixupdata(nn.Module):
    """Mixup for BYOL-A.

    Args:
        ratio: Alpha in the paper.
        n_memory: Size of memory bank FIFO.
        log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
    """

    def __init__(self, ratio=0.2, n_memory=2048, log_mixup_exp=True):
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []

    # @profile
    def forward(self, x):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))]
            # mix them
            mixed = log_mixup_exp(x, z, 1. - alpha) if self.log_mixup_exp \
                    else alpha * z + (1. - alpha) * x
        else:
            mixed = x
        # update memory bank
        self.memory_bank = (self.memory_bank + [x])[-self.n:]

        return mixed.to(torch.float)

# class Dataiemocap(Dataset):
#     def __init__(self, cfg, files_path, files_label, tfms):
#         # argment check
#         assert (files_label is None) or (len(files_path) == len(files_label)), 'The number of audio files and labels has to be the same.'
#         super().__init__()
#
#         # initializations
#         self.cfg = cfg
#         self.n = 1
#         self.c = 11
#         self.m = 16000
#         self.files = files_path
#         self.labels = files_label
#         self.tfms = tfms
#         # self.gender = gender_label
#         self.mel_spectrogram = T.MelSpectrogram(
#             n_fft=cfg.n_fft,
#             win_length=cfg.win_length,
#             hop_length=cfg.hop_length,
#             # normalized=True,
#             center=True,
#             pad_mode="reflect",
#             power=2.0,
#             norm='slaney',
#             onesided=True,
#             n_mels=cfg.n_mels,
#         )
#         self.delta_setting = T.ComputeDeltas()
#         # self.unit_length = int(cfg.unit_sec * cfg.sample_rate)
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         # load single channel .wav audio
#         wav, sr = torchaudio.load(self.files[idx])
#         wav = wav.mean(0, keepdims=True)
#
#         # resample
#         wav = T.Resample(sr, self.cfg.sample_rate)(wav)
#
#         assert wav.shape[0] == 1, f'Convert .wav files to single channel audio, {self.files[idx]} has {wav.shape[0]} channels.'
#         # (1, length) -> (length,)
#         wav = wav[0]
#
#         length_adj = 17000 - len(wav)
#         if length_adj > 0:
#             n = math.ceil(17000 / len(wav))
#             wav = torch.tile(wav, dims=(n,))[:17000]
#
#         num_shifts = self.n * self.c - 1
#
#         step_size = int(int(len(wav) - self.m) / num_shifts)
#         lms =[]
#         spl_data = []
#         start_idx = [0]
#         end_idx = [self.m]
#         for iii in range(num_shifts):
#             start_idx.extend([start_idx[0] + (iii + 1) * step_size])
#             end_idx.extend([end_idx[0] + (iii + 1) * step_size])
#             # Output Split Data
#         for iii in range(len(start_idx)):
#             spl_data.append(wav[start_idx[iii]: end_idx[iii]])
#         for j in spl_data:
#             mel_sp = (self.mel_spectrogram(j) + torch.finfo().eps).log()
#             mel_deltas = self.delta_setting(mel_sp)
#             mel_ddeltas = self.delta_setting(mel_deltas)
#             # mel_total = torch.stack((mel_sp,mel_deltas),dim=0)
#             mel_total = torch.stack((mel_sp,mel_deltas,mel_ddeltas),dim=0)
#             lms.append(mel_total)
#             # lms.append((self.mel_spectrogram(j) + torch.finfo().eps).log().unsqueeze(0))
#         total_lms = torch.stack(lms, dim=0)
#             # lms = self.to_melspecgram(wav).unsqueeze(0)
#
#             # # transform (augment)
#             # if self.tfms:
#             #     lms = self.tfms(lms)
#             # to log mel spectrogram -> (1, n_mels, time)
#             # lms = (self.mel_spectrogram(wav) + torch.finfo().eps).log()
#             # # lms = self.to_melspecgram(wav).unsqueeze(0)
#             # lmd1 = self.delta_setting(lms)
#             # lmd2 = self.delta_setting(lmd1)
#             #
#             # lms = torch.stack([lms,lmd1,lmd2], dim=0)
#             # transform (augment)
#         if self.tfms:
#             total_lms = self.tfms(total_lms)
#         # print(total_lms.shape)
#         # print(self.labels[idx])
#         if self.labels is not None:
#             return total_lms, torch.tensor(self.labels[idx],dtype=torch.float32)
#         return total_lms
class Dataiemocap(Dataset):
    def __init__(self, cfg, files_path, files_label, tfms,all_length = False):
        # argment check
        assert (files_label is None) or (len(files_path) == len(files_label)), 'The number of audio files and labels has to be the same.'
        super().__init__()

        # initializations
        self.cfg = cfg
        self.files = files_path
        self.sampling_rate = 16000
        self.labels = files_label
        # self.tfms = tfms
        # self.all_length = all_length
        # self.mel_spectrogram = T.MelSpectrogram(
        #     n_fft=cfg.n_fft,
        #     win_length=cfg.win_length,
        #     hop_length=cfg.hop_length,
        #     # normalized=True,
        #     center=True,
        #     pad_mode="reflect",
        #     power=2.0,
        #     norm='slaney',
        #     onesided=True,
        #     n_mels=cfg.n_mels,
        # )
        # self.delta_setting = T.ComputeDeltas()
        # self.unit_length = int(cfg.unit_sec * cfg.sample_rate)
        self.model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load single channel .wav audio
        wav, sr = torchaudio.load(self.files[idx])
        wav = wav.mean(0, keepdims=True)

        # resample
        wav = T.Resample(sr, self.cfg.sample_rate)(wav)

        assert wav.shape[0] == 1, f'Convert .wav files to single channel audio, {self.files[idx]} has {wav.shape[0]} channels.'
        # (1, length) -> (length,)
        wav = wav[0].detach().numpy()

        # self.mel_spectrogram.sample_rate = sr
        # if self.all_length:
        #     lms = (self.mel_spectrogram(wav) + torch.finfo().eps).log().unsqueeze(0)
        # else:
        #     # # zero padding to both ends
        #     length_adj = self.unit_length - len(wav)
        #     if length_adj > 0:
        #         half_adj = length_adj // 2
        #         wav = F.pad(wav, (half_adj, length_adj - half_adj))
        #
        #     # random crop unit length wave
        #     length_adj = len(wav) - self.unit_length
        #     start = random.randint(0, length_adj) if length_adj > 0 else 0
        #     wav = wav[start:start + self.unit_length]
        #
        #     # to log mel spectrogram -> (1, n_mels, time)
        #     lms = (self.mel_spectrogram(wav) + torch.finfo().eps).log().unsqueeze(0)
        #     # lms = self.to_melspecgram(wav).unsqueeze(0)
        #
        #     # # transform (augment)
        #     # if self.tfms:
        #     #     lms = self.tfms(lms)
        #     # to log mel spectrogram -> (1, n_mels, time)
        #     # lms = (self.mel_spectrogram(wav) + torch.finfo().eps).log()
        #     # # lms = self.to_melspecgram(wav).unsqueeze(0)
        #     # lmd1 = self.delta_setting(lms)
        #     # lmd2 = self.delta_setting(lmd1)
        #     #
        #     # lms = torch.stack([lms,lmd1,lmd2], dim=0)
        #     # transform (augment)
        # if self.tfms:
        #     lms = self.tfms(lms)

        res = {"feats": wav}
        if self.labels is not None:
            res["target"] = self.labels[idx]

        return res
        # if self.labels is not None:
        #     return lms, torch.tensor(self.labels[idx])
        # return lms

    def collator(self, samples):
        if len(samples) == 0:
            return {}
        feats = [s["feats"] for s in samples]
        inputs = self.processor(feats, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        # feats = [s["feats"] for s in samples]
        # sizes = [s.shape[0] for s in feats]
        labels = torch.tensor([s["target"] for s in samples]) if samples[0]["target"] is not None else None

        # target_size = max(sizes)
        #
        # collated_feats = feats[0].new_zeros(
        #     len(feats), target_size, feats[0].size(-1)
        # )
        #
        # padding_mask = torch.BoolTensor(torch.Size([len(feats), target_size])).fill_(True)
        # for i, (feat, size) in enumerate(zip(feats, sizes)):
        #     collated_feats[i, :size] = feat
        #     padding_mask[i, size:] = False
        #
        # attention_mask = padding_mask.init()
        # res = {
        #     "id": torch.LongTensor([s["id"] for s in samples]),
        #     "net_input": {
        #         "feats": collated_feats,
        #         "padding_mask": padding_mask
        #     },
        #     "labels": labels
        # }

        return inputs, labels

# def ASC_datasets(dataset, fold):
#     """Create Dataloaders for Fused"""
#
#     # Iterate the indicated train and val fold
#     # filenames = {}
#
#     with open('/home/songmeis/tianhao_byol/acoustic_scene_classification_add_ser/work/metadata_fused/train_fold_{}.txt'.format(fold)) as f:
#         names = f.readlines()
#     filenames = [x.strip() for x in names]
#
#     asc_filenames = [j.split(',')[1] for j in filenames]
#     # Draw from the Dataset according to the given train/val split
#     train_asc_indices = []
#
#     for i in range(len(dataset)):
#         if dataset.files[i] in asc_filenames:
#             train_asc_indices.append(i)
#
#     train_asc_dataset = torch.utils.data.Subset(dataset, train_asc_indices)
#     return train_asc_dataset
#
#
# def ASC_datasets_test(dataset, fold):
#     """Create Dataloaders for Fused"""
#
#     # Iterate the indicated train and val fold
#     # filenames = {}
#
#     with open('/home/songmeis/tianhao_byol/acoustic_scene_classification_add_ser/work/metadata_fused/test_fold_{}.txt'.format(fold)) as f:
#         names = f.readlines()
#     filenames = [x.strip() for x in names]
#
#     asc_filenames = [j.split(',')[0] for j in filenames]
#     # Draw from the Dataset according to the given train/val split
#     test_asc_indices = []
#
#     for i in range(len(dataset)):
#         if dataset.files[i] in asc_filenames:
#             test_asc_indices.append(i)
#
#     test_asc_dataset = torch.utils.data.Subset(dataset, test_asc_indices)
#     return test_asc_dataset
def IEMOCAP_fold_datasets(dataset):
    """Create Dataloaders for DCASE"""
    filenames = {}
    # Iterate the indicated train and test fold

    for fun in ['train_test', 'valid_test']:
    # for fun in ['train', 'valid']:
        with open('E:\\ser_add_ser\\msp_txt\\{}.txt'.format(fun)) as f:
            names = f.readlines()
        filenames[fun] = [x.strip() for x in names]
        # emo_filenames[fun] = [j.split(',')[0] for j in filenames[fun]]
    # Draw from the Dataset according to the given train/val split
    train_indices = []
    valid_indices = []

    for i in range(len(dataset)):
        if dataset.files[i] in filenames['train_test']:
            train_indices.append(i)

        if dataset.files[i] in filenames['valid_test']:
            valid_indices.append(i)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)

    return train_dataset, valid_dataset


def IEMOCAP_fold_datasets_test(dataset):
    """Create Dataloaders for DCASE"""
    # Iterate the indicated train and test fold
    with open('E:\\ser_add_ser\\msp_txt\\test_test.txt') as f:
    # with open('E:\\ser_add_ser\\msp_txt\\test.txt') as f:
        names = f.readlines()
    filenames = [x.strip() for x in names]
        # emo_filenames[fun] = [j.split(',')[0] for j in filenames[fun]]
    # Draw from the Dataset according to the given train/val split
    test_indices = []

    for i in range(len(dataset)):
        if dataset.files[i] in filenames:
            test_indices.append(i)

    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return test_dataset
