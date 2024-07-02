from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
import torch.nn as nn
from tqdm.auto import tqdm
import argparse
import sys
import torchvision.transforms as transforms
import os
from diffusers.optimization import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup
import wandb
import json
sys.path.insert(0, '../file')
from evaluator import evaluation_model


class iclevrDataset(Dataset):
    def __init__(self, root=None, mode="train"):
        super().__init__()
        if mode == 'train':
            with open('../file/train.json', 'r') as json_file:
                self.json_data = json.load(json_file)
            self.img_paths = list(self.json_data.keys())
            self.labels = list(self.json_data.values())
        elif mode == 'test':
            with open('../file/test.json', 'r') as json_file:
                self.labels = json.load(json_file)

        elif mode == 'new_test':
            with open('../file/new_test.json', 'r') as json_file:
                self.labels = json.load(json_file)
                
        self.labels_one_hot = []
        with open('../file/objects.json', 'r') as json_file:
            self.objects = json.load(json_file)
        for label in self.labels:
            label_one_hot = [0] * len(self.objects)
            for l in label:
                label_one_hot[self.objects[l]] = 1
            self.labels_one_hot.append(label_one_hot)
        self.labels_one_hot = torch.tensor(np.array(self.labels_one_hot))
    
        self.root = root   
        self.mode = mode
        self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
    def __len__(self):
        return len(self.labels)      
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            
            img_path = os.path.join(self.root, self.img_paths[idx])
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            
            return image, self.labels_one_hot[idx]
        elif self.mode == 'test':
            return self.labels_one_hot[idx]
        elif self.mode == 'new_test':
            return self.labels_one_hot[idx]
        
        
class CondUnet(nn.Module):
    def __init__(self, num_class=24, class_emb_size=4, embed_type='linear') -> None:
        super().__init__()
        # self.label_embedding = nn.Embedding(num_class, class_emb_size)
        self.embed_type = embed_type
        if self.embed_type == 'linear':
            self.label_embedding = nn.Linear(num_class, 64 * 64 * class_emb_size)
            self.model = UNet2DModel(
                sample_size=64,
                # in_channels=3+num_class * class_emb_size,
                in_channels=3 + class_emb_size,
                out_channels=3,
                time_embedding_type="positional",
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
                down_block_types=(
                    "DownBlock2D",  # a regular ResNet downsampling block
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        else:
            self.label_embedding = nn.Embedding(num_class, class_emb_size)
            self.model = UNet2DModel(
                sample_size=64,
                # in_channels=3+labels_num * embedding_label_size,
                in_channels=3 + num_class,
                out_channels=3,
                time_embedding_type="positional",
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
                down_block_types=(
                    "DownBlock2D",  # a regular ResNet downsampling block
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        self.class_emb_size = class_emb_size
        
    def forward(self, x, t, label):
        bs, c, w, h = x.shape
        if self.embed_type != 'linear': 
            '''nn Embedding'''
            embeded_label = label.view(bs, label.shape[1], 1, 1).expand(bs, label.shape[1], w, h)
            unet_input = torch.cat((x, embeded_label), 1)
            unet_output = self.model(unet_input, t).sample
            return unet_output
        else:
            '''Linear Embedding'''
            class_labels = label.float()
            class_cond = self.label_embedding(class_labels) # Map to embedding dimension
            class_cond = class_cond.view(bs, self.class_emb_size, w, h)
            # Net input is now x and class cond concatenated together along dimension 1
            net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)
            # print(f"net input shape: {net_input.shape}")
            # Feed this to the UNet alongside the timestep and return the prediction
            return self.model(net_input, t).sample # (bs, 1, 28, 28)


class ConditionlDDPM():
    def __init__(self, args):
        self.args = args
        
        self.device = args.device
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_train_timestamps = args.num_train_timestamps
        self.save_root = args.save_root
        self.label_embeding_size = args.label_embeding_size
        self.save_per_epoch = args.save_per_epoch
        self.embed_type = args.embed_type
        if args.noise_scheduler == 'cosine':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timestamps, beta_schedule="squaredcos_cap_v2")
        else:
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timestamps, beta_schedule="scaled_linear")
            
        self.noise_predicter = CondUnet(num_class=24, class_emb_size=self.label_embeding_size, embed_type=args.embed_type).to(self.device)
        self.eval_model = evaluation_model()
        
        self.train_dataset = iclevrDataset(root="../iclevr", mode="train")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # self.optimizer = torch.optim.Adam(self.noise_predicter.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.noise_predicter.parameters(), lr=self.lr)
        if args.lr_scheduler == 'cosine':
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=args.lr_warmup_steps,
                num_training_steps=len(self.train_dataloader) * self.epochs,
                num_cycles=50
            )
        elif args.lr_scheduler == 'constant':
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=args.lr_warmup_steps,
            )
        elif args.lr_scheduler == 'cosine-hard':
            self.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=args.lr_warmup_steps,
                num_training_steps=len(self.train_dataloader) * self.epochs,
                num_cycles=50
            )
        
    def train(self):
        best_test_acc = 0
        best_new_test_acc = 0
        loss_criterion = nn.MSELoss()
        # training 
        for epoch in range(1, self.epochs+1):
            epoch_loss = 0
            print(f"#################### epoch: {epoch}, lr {self.lr} ####################")
            for x, y in tqdm(self.train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                noise = torch.randn_like(x)
                timestamp = torch.randint(0, self.num_train_timestamps - 1, (x.shape[0], ), device=self.device).long()
                noise_x = self.noise_scheduler.add_noise(x, noise, timestamp)
                perd_noise = self.noise_predicter(noise_x, timestamp, y)
                
                
                loss = loss_criterion(perd_noise, noise)
                loss.backward()
                nn.utils.clip_grad_value_(self.noise_predicter.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.lr = self.lr_scheduler.get_last_lr()[0]
                epoch_loss += loss.item()
            print(f"Loss: {epoch_loss / len(self.train_dataloader):.4f}")
            test_acc = self.evaluate(epoch, target="test")
            new_test_acc = self.evaluate(epoch, target="new_test")
            # print(f"Test acc: {test_acc:.4f}, New Test acc: {new_test_acc:.4f}")
            wandb.log({
                "Loss": (epoch_loss / len(self.train_dataloader)),
                "DDPM Test Accuracy": test_acc,
                "DDPM New Test Accuracy": new_test_acc,
                "Learning Rate": self.lr_scheduler.get_last_lr()[0]
            })
            if (epoch == 1 or epoch % self.save_per_epoch == 0):
                if epoch == 1:
                    best_test_acc = test_acc
                    best_new_test_acc = new_test_acc
                self.save(os.path.join(self.args.ckpt_root, f"epoch={epoch}.ckpt"), epoch)
                
            elif test_acc > best_test_acc and new_test_acc > best_new_test_acc:
                best_test_acc = test_acc
                best_new_test_acc = new_test_acc
                self.save(os.path.join(self.args.ckpt_root, f"best_epoch={epoch}.ckpt"), epoch)
                
            elif test_acc > best_test_acc:
                best_test_acc = test_acc
                self.save(os.path.join(self.args.ckpt_root, f"best_test_epoch={epoch}.ckpt"), epoch)
                
            elif new_test_acc > best_new_test_acc:
                best_new_test_acc = new_test_acc
                self.save(os.path.join(self.args.ckpt_root, f"best_new_test_epoch={epoch}.ckpt"), epoch)
                
    def evaluate(self, epoch="final", target="test"):
        test_dataset = iclevrDataset(mode=f"{target}")
        test_dataloader = DataLoader(test_dataset, batch_size=32)
        for y in test_dataloader:
            y = y.to(self.device)
            x = torch.randn(32, 3, 64, 64).to(self.device)
            for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                with torch.no_grad():
                    pred_noise = self.noise_predicter(x, t, y)
                x = self.noise_scheduler.step(pred_noise, t, x).prev_sample
            acc = self.eval_model.eval(images=x.detach(), labels=y)
            denormalized_x = (x.detach() / 2 + 0.5).clamp(0, 1)
            print(f"* Accuracy of {target}.json on epoch {epoch}: {round(acc, 3)}" )
            generated_grid_imgs = make_grid(denormalized_x)
            save_image(generated_grid_imgs, f"{self.save_root}/{target}_{epoch}.jpg")
        return round(acc, 3)
    
    def save(self, path, epoch):
        torch.save({
            "noise_predicter": self.noise_predicter,
            "noise_scheduler": self.noise_scheduler,
            "optimizer": self.optimizer,
            "lr"        : self.lr,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "last_epoch": epoch
        }, path)
        print(f"save ckpt to {path}")
            
def main(args):
    conditionlDDPM = ConditionlDDPM(args)
    run = wandb.init(project="DLP_Lab6",
                     config=vars(args),
                     sync_tensorboard=True,
                     name=args.wandb_run_name,
                     save_code=True)
    args.tensorboard_log = f"runs/{args.wandb_run_name}"
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.ckpt_root, exist_ok=True)
    conditionlDDPM.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--device', type=str, choices=["cuda:0", "cuda:1", "cpu", "cuda"], default="cuda")
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--num_train_timestamps', type=int, default=1000)
    parser.add_argument('--lr-warmup_steps', default=0, type=int)
    parser.add_argument('--save-root', type=str, default="eval")
    parser.add_argument('--label-embeding_size', type=int, default=4)
    parser.add_argument('--save-per-epoch', type=int, default=10)
    # ckpt
    parser.add_argument('--ckpt-root', type=str, default="ckpt") # fro save
    parser.add_argument('--ckpt-path', type=str, default=None) # for load
    parser.add_argument('--wandb-run-name', type=str, required=False)
    parser.add_argument('--lr-scheduler', type=str, default='cosine')
    parser.add_argument('--noise-scheduler', type=str, default='cosine')
    parser.add_argument('--embed-type', type=str, default='linear')
    
    args = parser.parse_args()
    
    main(args)


