import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder, FeatureCombiner

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10
import wandb

import os
import torch


def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.current_epoch = current_epoch
        self.total_epochs = args.kl_anneal_cycle  # Total epochs for Monotonic or one cycle length for Cyclical
        self.annealing_type = args.kl_anneal_type
        self.max_beta = args.kl_anneal_ratio
        self.cycle_length = args.kl_anneal_cycle
        if self.annealing_type == 'NoAnnealing':
            self.beta = 1.0
        else:
            self.beta = 0.0

        
    def update(self):
        if self.annealing_type == 'Monotonic':
            # Monotonically increase beta from 0 to max_beta over total_epochs
            if self.current_epoch < self.total_epochs:
                self.beta = (self.current_epoch / self.total_epochs) * self.max_beta
            else:
                self.beta = self.max_beta
        elif self.annealing_type == 'Sinsoidal':
            # Cyclical annealing with sinusoidal variation
            cycle_position = (self.current_epoch % self.cycle_length) / self.cycle_length
            self.beta = self.max_beta * (np.sin(np.pi * cycle_position))
        elif self.annealing_type == 'NoAnnealing':
            # Beta is always max_beta (1.0), meaning full KL divergence impact from the start
            self.beta = self.max_beta
        elif self.annealing_type == 'Cyclical':
            cycle_position = (self.current_epoch % self.cycle_length) / self.cycle_length
            if cycle_position < 0.5:
                self.beta = self.max_beta * (cycle_position) * 2
            else:
                self.beta = self.max_beta 

        self.current_epoch += 1

    
    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        raise NotImplementedError
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.feature_combiner = FeatureCombiner(img_chans=128, label_chans=32, out_chans=128)
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        max_psnr = 0.0
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if self.tfr > 0 else False
            # print('data loader len =', len(train_loader))
            total_train_loss = 0.0
            for (img, label) in (pbar := tqdm(train_loader, ncols=200)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                total_train_loss += loss
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                
            val_loss, psnr = self.eval()
            if psnr > max_psnr:
                max_psnr = psnr
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}-best.ckpt"))
            elif self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            wandb.log({
                    "Train Loss": total_train_loss / len(train_loader),
                    "Validation Loss": val_loss,
                    "Beta": self.kl_annealing.get_beta(),
                    "psnr": psnr,
                    "Teacher Forcing Ratio": self.tfr
                })
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        total_loss = 0.0
        total_psnr = 0.0
        for (img, label) in (pbar := tqdm(val_loader, ncols=200)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0], psnr=psnr)
            total_loss += loss
            total_psnr += psnr
        return total_loss/len(val_loader), total_psnr/len(val_loader)
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        self.optim.zero_grad()
        batch_size, video_length, channels, height, width = img.size()
        step_loss = 0.0
        prev_frame = img[:, 0]
        for frame_idx in range(img.shape[1]):
            
            encoded_img = self.frame_transformation(img[:, frame_idx])
            encoded_label = self.label_transformation(label[:, frame_idx])
            z, mu, logvar = self.Gaussian_Predictor(encoded_img, encoded_label)
            if adapt_TeacherForcing and self.tfr == 1:
                decoded_features = self.Decoder_Fusion(encoded_img, encoded_label, z)
            elif adapt_TeacherForcing and self.tfr < 1:
                combined_features = self.tfr * encoded_img + (1 - self.tfr) * self.frame_transformation(prev_frame)
                decoded_features = self.Decoder_Fusion(combined_features, encoded_label, z)
            else:
                decoded_features = self.Decoder_Fusion(self.frame_transformation(prev_frame), encoded_label, z)
                
            output = self.Generator(decoded_features)
            
            prev_frame = output
            
            recon_loss = self.mse_criterion(output, img[:, frame_idx])
            kl_loss = kl_criterion(mu, logvar, batch_size)
            beta = self.kl_annealing.get_beta()
            total_loss = recon_loss + beta * kl_loss
            # Backward and optimize
            step_loss += total_loss
        step_loss /= img.shape[1]
        step_loss.backward()                   
        self.optimizer_step()
        return step_loss
    
    def val_one_step(self, img, label):
        with torch.no_grad():
            ret_loss = 0.0
            psnr = 0.0
            prev_frame = img[:, 0]
            for frame_idx in range(img.shape[1]):
                batch_size, video_length, channels, height, width = img.size()
                encoded_img = self.frame_transformation(prev_frame)
                encoded_label = self.label_transformation(label[:, frame_idx])
                z, mu, logvar = self.Gaussian_Predictor(encoded_img, encoded_label)
                # z = mu + torch.exp(logvar / 2) * torch.randn_like(logvar)
                decoded_features = self.Decoder_Fusion(encoded_img, encoded_label, z)
                output = self.Generator(decoded_features)
                
                prev_frame = output
                
                recon_loss = self.mse_criterion(output, img[:, frame_idx])
                kl_loss = kl_criterion(mu, logvar, batch_size)
                beta = self.kl_annealing.get_beta()
                total_loss = recon_loss + beta * kl_loss
                ret_loss += total_loss
                psnr += Generate_PSNR(img[:, frame_idx], output)

            return ret_loss/img.shape[1], psnr/img.shape[1]
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.args.tfr_sde:
            self.tfr -= self.args.tfr_d_step
            self.tfr = max(0, self.tfr)
            
    def tqdm_bar(self, mode, pbar, loss, lr, psnr=None):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        if mode == 'val':
            pbar.set_postfix(loss=float(loss), PSNR=float(psnr), refresh=False)
        else:
            pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,   default=None, help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    
    parser.add_argument('--wandb_run_name', type=str, default=None, help="Wandb run name")
    

    args = parser.parse_args()
    
    run = wandb.init(project="DLP_Lab4",
                     config=vars(args),
                     sync_tensorboard=True,
                     name=args.wandb_run_name,
                     save_code=True, )
    args.tensorboard_log = f"runs/{args.wandb_run_name}"
    main(args)
