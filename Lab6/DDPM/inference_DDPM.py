from DDPM import ConditionlDDPM, CondUnet
import torch
import os
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup
from torchvision.utils import save_image, make_grid
from tqdm.auto import tqdm

import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--device', type=str, choices=["cuda:0", "cuda:1", "cpu", "cuda"], default="cuda")
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--num_train_timestamps', type=int, default=1000)
    parser.add_argument('--lr_warmup_steps', default=0, type=int)
    parser.add_argument('--save-root', type=str, default="eval")
    parser.add_argument('--label_embeding_size', type=int, default=4)
    parser.add_argument('--save-per-epoch', type=int, default=10)
    # ckpt
    parser.add_argument('--ckpt_root', type=str, default="ckpt") # fro save
    parser.add_argument('--ckpt_path', type=str, default=None) # for load
    parser.add_argument('--wandb-run-name', type=str, required=False)
    parser.add_argument('--lr-scheduler', type=str, default='cosine')
    parser.add_argument('--embed-type', type=str, default='linear')
    parser.add_argument('--noise-scheduler', type=str, default='cosine')
    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    model = ConditionlDDPM(args) 
    checkpoint = torch.load(model.args.ckpt_path)
    model.noise_predicter = checkpoint["noise_predicter"]
    model.noise_scheduler = checkpoint["noise_scheduler"]
    model.optimizer = checkpoint["optimizer"]
    model.lr = checkpoint["lr"]
    model.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    model.epoch = checkpoint['last_epoch']
    
    model.evaluate(target="test")
    model.evaluate(target="new_test")
    
    label_one_hot = [0] * 24
    label_one_hot[9] = 1
    label_one_hot[7] = 1
    label_one_hot[22] = 1
    label_one_hot = torch.tensor(label_one_hot).to(model.device)
    label_one_hot = torch.unsqueeze(label_one_hot, 0)
    # breakpoint()
    x = torch.randn(1, 3, 64, 64).to(args.device)
    img_list = []
    for i, t in tqdm(enumerate(model.noise_scheduler.timesteps)):
        with torch.no_grad():
            pred_noise = model.noise_predicter(x, t, label_one_hot)
        x = model.noise_scheduler.step(pred_noise, t, x).prev_sample
        if(t % 50 == 0):
            denormalized_x = (x.detach() / 2 + 0.5).clamp(0, 1)
            img_list.append(denormalized_x)
    grid_img = make_grid(torch.cat(img_list, dim=0), nrow=5)
    save_image(grid_img, f"{model.args.save_root}/diffusion_process.jpg")
    