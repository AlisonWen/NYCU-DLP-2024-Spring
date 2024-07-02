import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import wandb


class WarmupLinearLRSchedule:
    """
    Implements Warmup learning rate schedule until 'warmup_steps', going from 'init_lr' to 'peak_lr' for multiple optimizers.
    """
    def __init__(self, optimizer, init_lr, peak_lr, end_lr, warmup_epochs, epochs=100, current_step=0):
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.optimizer = optimizer
        if warmup_epochs != 0:
            self.warmup_rate = (peak_lr - init_lr) / warmup_epochs
        else:
            self.warmup_rate = 0
        # print(f"end_lr: {end_lr}, peak_lr: {peak_lr}, epochs: {epochs}, warmup_epochs: {warmup_epochs}")
        self.decay_rate = (end_lr - peak_lr) / (epochs - warmup_epochs)
        self.update_steps = current_step
        self.lr = init_lr
        self.warmup_steps = warmup_epochs
        self.epochs = epochs
        if current_step > 0:
            self.lr = self.peak_lr + self.decay_rate * (current_step - 1 - warmup_epochs)

    def set_lr(self, lr):
        print(f"Setting lr: {lr}")
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def step(self):
        if self.update_steps <= self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
        # elif self.warmup_steps < self.update_steps <= self.epochs:
        else:
            lr = max(0., self.lr + self.decay_rate)
        self.set_lr(lr)
        self.lr = lr
        self.update_steps += 1
        return self.lr


#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers(args)
        self.prepare_training()
        self.device = args.device
        self.current_epoch = 0
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader, cur_epoch, total_epoch, cluster_mask=False):
        ret_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {cur_epoch}")):
            inputs = data.to(args.device)

            logits, targets = self.model(inputs, cur_epoch, total_epoch, cluster_mask)

            # Flatten the outputs and targets to fit cross-entropy expectation
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()

            self.optim.step()
            self.optim.zero_grad()

            ret_loss += loss.item()

        average_loss = ret_loss / len(train_loader)
        if args.scheduler == 'warm-up':
            lr = trainer.scheduler.lr
        else:
            lr = trainer.scheduler.get_last_lr()[0]
        print(f"Average training loss for epoch {cur_epoch}: {average_loss:.4f}, lr: {lr}")
        return average_loss
        
    def eval_one_epoch(self, val_loader, cur_epoch, total_epoch, cluster_mask):
        ret_loss = 0
        total_samples = 0
        
        with torch.no_grad():  # No gradients needed for validation, saves memory and computations
            for data in tqdm(val_loader, desc=f"Validating Epoch {cur_epoch}"):
                inputs = data.to(self.device)

                logits, targets = self.model(inputs, cur_epoch, total_epoch, cluster_mask)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

                ret_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        average_loss = ret_loss / len(val_loader)
        print(f"Validation Loss for Epoch {cur_epoch}: {average_loss:.4f}")
        return average_loss

    def configure_optimizers(self, args):
        if args.scheduler == 'warm-up':
            optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=1e-4, betas=(0.9, 0.96), weight_decay=4.5e-2)
            scheduler = WarmupLinearLRSchedule(
                optimizer=optimizer,
                init_lr=1e-6,
                peak_lr=args.learning_rate,
                end_lr=0.,
                warmup_epochs=10,
                epochs=args.epochs,
                current_step=args.start_from_epoch
            )
        else:
            optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5], gamma=0.1)
        return optimizer,scheduler
    def save(self, path):
        torch.save(self.model.transformer.state_dict(), path + '.ckpt')
        print(f"save ckpt to {path}.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="../dataset/lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="../dataset/lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--scheduler', type=str, default='warm-up', help='warm-up or multi-step')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')
    
    parser.add_argument('--wandb_run_name', type=str, required=False)
    parser.add_argument('--sos-token', type=int, default=1025, help='Start of Sentence token.')
    parser.add_argument('--cluster-mask', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    
    run = wandb.init(project="DLP_Lab5",
                     config=vars(args),
                     sync_tensorboard=True,
                     name=args.wandb_run_name,
                     save_code=True, )
    args.tensorboard_log = f"runs/{args.wandb_run_name}"
#TODO2 step1-5:    
    os.makedirs(args.save_root, exist_ok=True)
    trainer = TrainTransformer(args, MaskGit_CONFIGS)
    min_eval_loss = float(1e9+7)
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = trainer.train_one_epoch(train_loader, epoch, args.epochs, False)
        print(f"Epoch {epoch}: Training Loss: {train_loss:.4f}")
        eval_loss = trainer.eval_one_epoch(val_loader, epoch, args.epochs, False)
        if eval_loss < min_eval_loss:
            trainer.save(args.save_root + "/best-epoch=" + str(epoch))
            min_eval_loss = eval_loss
        elif epoch % args.save_per_epoch == 0:
            trainer.save(args.save_root + "/epoch=" + str(epoch))
        trainer.scheduler.step()    
        if args.scheduler == 'warm-up':
            lr = trainer.scheduler.lr
        else:
            lr = trainer.scheduler.get_last_lr()[0]
        wandb.log({
            "Train Loss": train_loss,
            "Valid Loss": eval_loss,
            "Gamma": trainer.model.gamma(epoch / args.epochs),
            "Learning Rate": lr
        })    
        trainer.current_epoch += 1
        