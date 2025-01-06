import os, sys
import argparse
import copy
import ipdb
from datetime import datetime

from pydub import AudioSegment
import librosa
import scipy

import torch
import torchaudio
from torch.utils.data import Dataset

# Local Imports
import utils

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

torch.random.manual_seed(2667)

def dataset_splits():
    root_path = "data/diplomacy_lines"
    sr = 44100
    # Audio 
    wav_path = os.path.join(root_path, "diplomacy_lines.wav")
    audio, sr = librosa.load(wav_path, sr=sr)
    breakpoint()
    audio = librosa.stft(audio, hop_length=0.0125, win_length=0.05, window="hann")
    #audio = scipy.signal.stft(audio, )
    audiodb = librosa.amplitude_to_db(abs(audio))

    # Text
    text_path = os.path.join(root_path, "ikit_diplomacy_subs.txt")
    with open(text_path, "r") as f:
        lines = f.readlines()
    lines = lines[1:]
    data = []
    start = 0
    for x in range(0, len(lines), 2):
        text = lines[x]
        text = utils.remove_endline(text)
        data_dict = {'text':text}
        end = utils.remove_endline(lines[x+1])
        end = datetime.strptime(end,'%M:%S')
        end = end.second + end.minute*60
        data_dict['start'] = start
        data_dict['end'] = end
        data_dict['audio'] = audio[sr*start:sr*end]
        data_dict['gt_waveform'] = None
        start = end
        data.append(data_dict)

    train = data
    valid = [data[0]]
    print("\n\nTODO Custom validatons: Manually make this. Chief warlock ikit claw. Knows who his boss is. What can clan scryer do for you?\n\n")
    return train, valid


class GeneralDataset(Dataset):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        # Create train, valid
        self.train, self.valid = dataset_splits()
        #if args.shuffle:
        #    self.train, self.valid = utils.split_container_by_ratio(self.train+self.valid+self.test, (len(self.train), len(self.valid), seed=shuffle_random_seed)
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        self.processor = bundle.get_text_processor()


    def choose_split(self, split):
        if split == "train":
            self.data = self.train
            del self.valid
        if split == "valid":
            self.data = self.valid
            del self.train


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        data = self.data[idx]
        text = data['text']
        inputs, lengths = self.processor(text)
        batch = {"inputs":inputs, "lengths":lengths}
        return batch


class GeneralDatasetPreloaded(GeneralDataset):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def choose_split(self, split):
        self.all_data = []
        if split == "train":
            self.data = self.train
            del self.valid
        if split == "valid":
            self.data = self.valid
            del self.train
        for idx in tqdm(range(self.__len__()), total=self.__len__(), desc=f"Preloading {split} Dataset"):
            data = super().__getitem__(idx)
            self.all_data.append(data)

    def __getitem__(self, idx):
        return self.all_data[idx]
