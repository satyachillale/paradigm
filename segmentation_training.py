
import argparse
import glob
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

from UNet.unet import UNet
from unet_invoke import train_model

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

dataset_dir = "../dataset"

def get_a_random_image(dir=dataset_dir + "/train"):
    all_videos = os.listdir(dir)
    picked_video = random.choice(all_videos)
    all_images = os.listdir(os.path.join(dir, picked_video))
    picked_image = random.choice(all_images)
    picked_image_idx = int(picked_image[len("image_"):-len(".png")])

    img = Image.open(os.path.join(dir, picked_video, picked_image))
    data = np.asarray(img, dtype="int32")

    mask = np.load(os.path.join(dir, picked_video, "mask.npy"))
    return data, mask[picked_image_idx]

"""## Plotting images to visualize more masked frames"""
def visualize_labels(dir=dataset_dir + "/train", num_frames=5):
    all_videos = os.listdir(dir)
    picked_video = random.choice(all_videos)
    video_path = os.path.join(dir, picked_video)
    all_images = sorted([img for img in os.listdir(video_path) if img.startswith('image_')])
    picked_images = random.sample(all_images, num_frames)

    mask = np.load(os.path.join(video_path, "mask.npy"))

    plt.figure(figsize=(15, 3 * num_frames))

    for i, img_name in enumerate(picked_images):
        img_path = os.path.join(video_path, img_name)
        img = Image.open(img_path)
        print("img size: ", img.size, '\n')
        data = np.asarray(img, dtype="int32")
        frame_idx = int(img_name[len("image_"):-len(".png")])

        plt.subplot(num_frames, 2, 2*i + 1)
        plt.imshow(data)
        plt.title(f"Frame {frame_idx}")
        plt.axis('off')

        plt.subplot(num_frames, 2, 2*i + 2)
        plt.imshow(mask[frame_idx])
        plt.title(f"Mask for Frame {frame_idx}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

"""Call the function to visualize labels"""

def visualize():
    print(f"The training set has {len(os.listdir(dataset_dir + '/train'))} videos, and each video has {len(glob.glob(dataset_dir + '/train/video_00000/*.png'))} frames and {len(glob.glob(dataset_dir + '/train/video_00000/*.npy'))} mask file for all frames")
    print(f"The validation set has {len(os.listdir(dataset_dir + '/val'))} videos, and each video has {len(glob.glob(dataset_dir + '/val/video_01000/*.png'))} frames and {len(glob.glob(dataset_dir + '/val/video_01000/*.npy'))} mask file for all frames")
    print(f"The unlabeled set has {len(os.listdir(dataset_dir + '/unlabeled'))} videos, and each video has {len(glob.glob(dataset_dir + '/unlabeled/video_02000/*.png'))} frames and {len(glob.glob(dataset_dir + '/unlabeled/video_02000/*.npy'))} mask file for all frames")

    mask = np.load(dataset_dir + '/train/video_00000/mask.npy')
    print(f"The mask file has the shape {mask.shape}.")
    img = Image.open(dataset_dir + '/train/video_00000/image_0.png')
    data = np.asarray(img, dtype="int32")
    print(f"Each image has the shape {data.shape}")

    print("Let's visualize them:")

    data, mask = get_a_random_image()
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.imshow(data)
    plt.axis('off')

    # Plot the second image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.imshow(mask)
    plt.axis('off')

    # visualize_labels(f"smalldataset/dataset/train", num_frames=10)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(device)
    model = UNet(n_channels=3, n_classes=49, bilinear=False)
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)
    train_model(
        model=model,
        dataset_dir=dataset_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        amp=args.amp
    )
    torch.save(model.state_dict(), "paradigm_segmentation.pth")