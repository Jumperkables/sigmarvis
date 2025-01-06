# Please help me set up the basics of a pytorch dataset class for loading audio data
# 1. Read from ./data
# 2. Recursively search for all .wav files in various subdirectories
# 3. Also collect the accompanying .json files that contain the transcriptions
# 4. The transcript is from the 'text' key in the json file
# 5. Implement the __len__ and __getitem__ 

# standard imports
import os
import json

# 3rd party imports
import torch
from torch.utils.data import Dataset

class SigmarAudioDataset(Dataset):
    def __init__(self, root_dir='../data/_Saltzpyre_1bd6b13e01e224f5.stream_24000'):
        self.root_dir = root_dir
        self.data = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_path = os.path.join(root, file)
                    json_path = wav_path.replace('.wav', '.json')
                    with open(json_path, 'r') as f:
                        transcript = json.load(f)['text']
                    self.data.append((wav_path, transcript))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wav_path, transcript = self.data[idx]
        return wav_path, transcript
    
if __name__ == '__main__':
    dataset = SigmarAudioDataset()
    print(len(dataset))
    print(dataset[0])