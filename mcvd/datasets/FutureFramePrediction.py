import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import re

class FutureFramePrediction(Dataset):
    def __init__(self, rootDir, mode, split='train', transforms=None):
        self.videoFolders = []
        self.mode = mode
        self.dataDir = os.path.join(rootDir, split)
        self.split = split
        self.framesPerVideo = 11
        self.transforms = transforms
        for folder in os.listdir(self.dataDir):
            if os.path.isdir(os.path.join(self.dataDir, folder)):
                self.videoFolders.append(os.path.join(self.dataDir, folder))
    
    def __len__(self):
        if self.mode == 'last':
            return len(self.videoFolders)
        return len(self.videoFolders) * self.framesPerVideo
    
    def __getitem__(self, idx):        
        videoIndex = idx
        startFrameIdx = 0
        
        requestedFrameIdxs = [startFrameIdx + i for i in range(self.framesPerVideo)]
        
        if self.mode == 'last':
            requestedFrameIdxs = [startFrameIdx + i for i in range(self.framesPerVideo * 2)]
        else:
            requestedFrameIdxs.append(0)
        
        frames = []
        pattern = re.compile(r'video_(\d+)$')
        match = pattern.search(self.videoFolders[videoIndex])
        videoNumber = int(match.group(1))
        
        for i in requestedFrameIdxs:
            imgPath = os.path.join(self.videoFolders[videoIndex], f"image_{i}.png")
            image = Image.open(imgPath)
            
            if self.transforms:
                image = self.transforms(image)
            frames.append(image)
        
        return torch.stack(frames), torch.tensor(videoNumber)


    











