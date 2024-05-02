import os
import torch
from UNet.unet import UNet as unet_model
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import re, sys
import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"


class PredictionData(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.transform = transforms.Compose([transforms.Resize((160,240)),transforms.ToTensor()])
        self.images = os.listdir(dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        match = re.search(r'pred_(\d+).png$', self.images[idx])
        video_number = int(match.group(1))
        image = self.transform(image)
        return image, video_number


def predict(ckpt, data_path):
    model = unet_model(3, 49, False).to(device)
    m = torch.load(ckpt)
    model.load_state_dict(m)
    model.eval()
    preds = PredictionData(dir=data_path)
    predLoader = DataLoader(preds, batch_size=10, shuffle=False)
    results = {}
    for i, (image, video_number) in enumerate(tqdm(predLoader, desc="Generating Masks for predictions")):
        image = image.to(device)
        with torch.no_grad():
            output = model(image)
            preds = torch.argmax(output, axis=1)
        for j in range(video_number.shape[0]):
            results[video_number[j].item()] = preds[j].unsqueeze(0)
    return results


if __name__ == "__main__":
    
    unetModelPath = sys.argv[1]
    preds = sys.argv[2]
    masks = predict(unetModelPath,preds)
    video_nums = list(masks.keys())
    video_nums.sort()
    result = []
    for num in video_nums:
        result.append(masks[num].to("cpu"))
    result = torch.concat(result, dim = 0)
    print("final shape",result.shape)
    torch.save(result, "paradigm_segmentation_results.pt")
