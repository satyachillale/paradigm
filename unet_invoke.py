import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from UNet.utils.dice_score import dice_loss
from UNet.evaluate import evaluate

class TrainingData(Dataset):

    def __init__(self, data):
        self.images, self.masks = [], []
        for i in data:
            imgs = os.listdir(i)
            for img in imgs:
                if not img.startswith('mask'):
                    self.images.append(i + '/' + img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]))/255
        try:
          x = self.images[idx].split('/')
          image_name = x[-1]
          mask_idx = int(image_name.split("_")[1].split(".")[0])
          x = x[:-1]
          mask_path = '/'.join(x)
          mask = np.load(mask_path + '/mask.npy')
          mask = mask[mask_idx, :, :]
        except:
          mask = np.zeros((160, 240))
        return img, mask
    
class UnlabeledData(Dataset):

    def __init__(self, videos, transform=None):
        self.transforms = transform
        self.images, self.masks = [], []
        for i in videos:
            imgs = os.listdir(i)
            self.images.extend([i + '/' + img for img in imgs if not img.startswith('mask')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]))/255
        return img

def train_model(
        model,
        dataset_dir,
        device,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    train_data_path = os.path.join(dataset_dir,'train/video_') #Change this to your train set path
    val_data_path = os.path.join(dataset_dir,'val/video_') #Change this to your validation path

    train_data_dir = [train_data_path + f"{i:05d}" for i in range(0, 1000)]
    train_data = TrainingData(train_data_dir)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data_dir = [val_data_path + f"{i:05d}" for i in range(1000, 2000)]
    val_data = TrainingData(val_data_dir)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_data), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for idx, (data, targets) in enumerate(train_dataloader):
                data = data.permute(0, 3, 1, 2)
                images, true_masks = data, targets.type(torch.long).to(device)
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

            val_score = evaluate(model, val_dataloader, device, amp)
            print(f"Epoch: {epoch}, dice-score: {val_score}")
            scheduler.step(val_score)


def test(model, dataset_dir, batch_size, device):
    test_data_path = os.path.join(dataset_dir,'train/video_') #Change this to your train set path
    test_data_dir = [test_data_path + f"{i:05d}" for i in range(901, 1000)]
    test_data = TrainingData(test_data_dir, None)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    for idx, (data, targets) in enumerate(test_dataloader):
        data = data.permute(0, 3, 1, 2)
        images, true_masks = data, targets.type(torch.long).to(device)
        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        output = model(images)
        pred_mask = torch.argmax(F.softmax(output), dim=1)
        plt.figure(figsize=(10, 5))

        # plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        # plt.imshow(data[0])
        # plt.axis('off')

        # Plot the second image
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 2nd subplot
        plt.imshow(true_masks[0].cpu().numpy())
        plt.axis('off')

        # Plot the second image
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        plt.imshow(pred_mask[0].cpu().numpy())
        plt.axis('off')

def generate_unlabeled_mask(model, dataset_dir, batch_size, device):
    unlabeled_data_path = os.path.join(dataset_dir,'unlabeled/video_') #Change this to your train set path
    unlabeled_data_dir = [unlabeled_data_path + f"{i:05d}" for i in range(0, 1000)]
    unlabeled_data = UnlabeledData(unlabeled_data_dir, None)
    dataloader = torch.utils.data.DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True)
    for idx, (data) in enumerate(dataloader):
        images = data.permute(0, 3, 1, 2)
        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        output = model(images)
        pred_mask = torch.argmax(F.softmax(output), dim=1)
        np.save(os.path.join("../unlabeled_masks", '{idx}_mask.npy'), pred_mask)

        
