import numpy as np
import os
import pickle
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# class ParadigmDataset(Dataset):

#     def __init__(self, data_dir, frames_per_sample=5, train=True, random_time=True, random_horizontal_flip=True,
#                  total_videos=-1, with_target=True, start_at=0):

#         self.data_dir = data_dir
#         self.train = train
#         self.frames_per_sample = frames_per_sample
#         self.random_time = random_time
#         self.random_horizontal_flip = random_horizontal_flip
#         self.total_videos = total_videos
#         self.with_target = with_target
#         self.start_at = start_at

#         self.train_dir = data_dir
#         if self.train:
#             self.train_video_dirs = [os.path.join(self.train_dir, f'video_{str(i).zfill(5)}') for i in range(2000, 2100)]
#         else:
#             self.train_video_dirs = [os.path.join(self.train_dir, f'video_{str(i).zfill(5)}') for i in range(1000, 1100)]
#         print(f"Dataset length: {self.__len__()}")

#     def len_of_vid(self, index):
#         video_index = index % self.__len__()
#         video_dir = self.train_video_dirs[video_index]
#         return len(os.listdir(video_dir))

#     def __len__(self):
#         return self.total_videos if self.total_videos > 0 else len(self.train_video_dirs)

#     def max_index(self):
#         return len(self.train_video_dirs)

#     def __getitem__(self, index, time_idx=0):

#         # Use `index` to select the video directory
#         video_dir = self.train_video_dirs[index]
#         frames = []

#         # Collect frames from the video directory
#         for filename in sorted(os.listdir(video_dir)):
#             if filename.endswith('.png'):
#                 img_path = os.path.join(video_dir, filename)
#                 img = Image.open(img_path)
#                 frames.append(transforms.ToTensor()(img))

#         # Randomly choose a window of frames if random_time is enabled
#         video_len = len(frames)
#         if self.random_time and video_len > self.frames_per_sample:
#             time_idx = np.random.randint(video_len - self.frames_per_sample)
#         time_idx += self.start_at
#         prefinals = frames[time_idx:time_idx+self.frames_per_sample]

#         # Apply random horizontal flip if enabled
#         flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
#         prefinals = [transforms.RandomHorizontalFlip(flip_p)(img) for img in prefinals]

#         if self.with_target:
#             return torch.stack(prefinals), torch.tensor(1)
#         else:
#             return torch.stack(prefinals)
class ParadigmDataset(Dataset):
    def __init__(self, root_dir, split = 'train', mode = 'cont', tranforms = None):
        self.map_idx_image_folder = []
        self.mode = mode
        self.data_dir = os.path.join(root_dir, split)
        self.split = split
        self.per_vid_data_len = 11
        self.transforms = tranforms
        for v in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, v)):
                self.map_idx_image_folder.append(os.path.join(self.data_dir, v))
    
    def __len__(self):
        if self.mode == 'last':
            return len(self.map_idx_image_folder  )
        return len(self.map_idx_image_folder  ) * self.per_vid_data_len
    
    def __getitem__(self, idx):
        # if self.split == "train": # return initital 11 frame only
        video_num = idx // self.per_vid_data_len
        start_idx = idx % self.per_vid_data_len
        if self.mode == 'last':
            video_num = idx
            start_idx = 0
        
        req_image_idx= [start_idx + i for i in range(0,11)]

        if self.mode == 'last':
            #req_image_idx.append(21)
            req_image_idx= [start_idx + i for i in range(0,22)]
        else:
            req_image_idx.append(start_idx + 11) # add 12 th frame

        images = []
        pattern = re.compile(r'video_(\d+)$')
        #video_number = int(match.group(1))
        match = pattern.search(self.map_idx_image_folder[video_num])
        video_number = int(match.group(1))
        for i in req_image_idx:
            img_path = os.path.join(self.map_idx_image_folder[video_num], f"image_{i}.png" )
            image = Image.open(img_path)

            if self.transforms:
                image = self.transforms(image)
            images.append(image)

        return torch.stack(images), torch.tensor(video_number)