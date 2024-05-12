from load_model_from_ckpt import load_model, get_sampler, init_samples
from datasets import get_dataset, data_transform, inverse_data_transform
from runners.ncsn_runner import conditioning_fn
from models import ddpm_sampler

import glob, os
import torch
from torch.utils.data import DataLoader, Dataset
import sys
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import re
from PIL import Image
from functools import partial
from tqdm import tqdm
from datasets.FutureFramePrediction import FutureFramePrediction

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict_one_frame_autoregressive(dataDir, modelPath, outDir, split):
    scorenet, config = load_model(modelPath, device)
    sampler = partial(ddpm_sampler, config=config)
    test_transform = transforms.Compose([
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor()
            ])
    hiddenDataset = FutureFramePrediction(rootDir=dataDir, split=split, mode= 'none', transforms= test_transform )
    print("Hidden data size",len(hiddenDataset))
    hiddenDataloader = DataLoader(hiddenDataset, batch_size=200, shuffle=False,
                         num_workers=config.data.num_workers, drop_last=False)

    avg_mse = 0.0
    
    for i, (frames, video_num) in enumerate(hiddenDataloader):
        frames = data_transform(config, frames)
        real, cond, cond_mask = conditioning_fn(config, frames, num_frames_pred=config.data.num_frames,
                                        prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
                                        prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0))

        real = inverse_data_transform(config, real)
        cond = cond.to(config.device)
        init_samples_shape = (real.shape[0], config.data.channels*config.data.num_frames,
                                  config.data.image_size, config.data.image_size)
        init_samples = torch.randn(init_samples_shape, device=config.device)
        n_iter_frames = 11

        pred_samples = []
        last_frame = None
        for i_frame in tqdm(range(n_iter_frames), desc="Generating video frames"):
            gen_samples = sampler(init_samples,
                                        scorenet, cond=cond, cond_mask=cond_mask,
                                        n_steps_each=config.sampling.n_steps_each,
                                        step_lr=config.sampling.step_lr,
                                        verbose=True,
                                        final_only=True,
                                        denoise=config.sampling.denoise,
                                        subsample_steps=1000,
                                        clip_before=getattr(config.sampling, 'clip_before', True),
                                        t_min=getattr(config.sampling, 'init_prev_t', -1),
                                        gamma=getattr(config.model, 'gamma', False))

            gen_samples = gen_samples[-1].reshape(gen_samples[-1].shape[0], config.data.channels*config.data.num_frames,
                                                        config.data.image_size, config.data.image_size)
            pred_samples.append(gen_samples.to('cpu'))
            last_frame = gen_samples.to('cpu')
            if i_frame == n_iter_frames - 1:
                last_frame = gen_samples.to('cpu')
                continue

            cond = torch.cat([cond[:, config.data.channels:], gen_samples[:, :config.data.channels]], dim=1)
            init_samples = torch.randn(init_samples_shape, device=config.device)

        pred = torch.cat(pred_samples, dim=1)
        pred = inverse_data_transform(config, pred)
        last_frame = inverse_data_transform(config, last_frame)
        out_folder_name = os.path.join(outDir, f"predicted_images")
        if not os.path.exists(out_folder_name):
          os.mkdir(out_folder_name)
        for idx in range(video_num.shape[0]):
          filename = f"pred_{int(video_num[idx])}.png"
          save_image(last_frame[idx], os.path.join(out_folder_name,filename))
          print(f"Saved video {int(video_num[idx])} to {os.path.join(out_folder_name,filename)}")

if __name__ == "__main__":
    
    dataDir = sys.argv[1]
    modelPath = sys.argv[2]
    outDir = sys.argv[3]
    predict_one_frame_autoregressive(dataDir=dataDir, modelPath=modelPath,outDir=outDir,split='hidden')





