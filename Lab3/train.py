import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, '../')
from src.models.resnet34_unet import resnet34_unet
from src.models.unet import UNet
# from src.models.gpt_unet import UNet
# from sample_unet import UNet
from src.oxford_pet import SimpleOxfordPetDataset, load_dataset
from src.utils import dice_loss, calculate_accuracy
# from src.utils import MixedLoss, DiceLoss

# Assume UNet and other required imports are already defined above

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device, dtype=torch.float32)
        masks = batch['mask'].to(device, dtype=torch.float32)
        
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = dice_loss(F.sigmoid(outputs), masks)
        accuracy = calculate_accuracy(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_accuracy += accuracy.item() * images.size(0)
        if batch_idx % 10 == 0:
            print(f"Train : [{batch_idx * len(batch['image'])}/{len(dataloader.dataset)}"
      f" ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = running_accuracy / len(dataloader.dataset)
    return epoch_loss, epoch_accuracy

def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device, dtype=torch.float32)
            masks = batch['mask'].to(device, dtype=torch.float32)
            
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            accuracy = calculate_accuracy(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            running_accuracy += accuracy.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = running_accuracy / len(dataloader.dataset)
    return epoch_loss, epoch_accuracy

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model', help='unet or resnet34_unet')
    parser.add_argument('--log_path', help="Tensorboard log path name", type=str, default=None)
    parser.add_argument('--save_as', help="The name of the saved model", type=str, default=None)
    parser.add_argument('--epoch', help="Number of epochs", type=int, default=None)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.log_path is not None:
        writer = SummaryWriter(args.log_path)
    else:
        writer = SummaryWriter('./UNet-1')
    # Parameters
    data_path = "../dataset"
    batch_size = 4
    num_epochs = 50
    if args.epoch is not None:
        num_epochs = args.epoch
    learning_rate = 0.001
    
    # Model
    if args.use_model == 'unet':
        model = UNet(n_channels=3, n_classes=1).to(device)
        print('Using UNet')
    else:
        model = resnet34_unet(num_classes=1).to(device)
        print('Using ResNet34 + UNet')
    
    save_as_model = "unet.pth" 
    if args.save_as is not None:
        save_as_model = args.save_as + ".pth"

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load Data
    train_loader = load_dataset(data_path, mode='train', batch_size=batch_size)
    valid_loader = load_dataset(data_path, mode='valid', batch_size=batch_size)
    
    # Training Loop
    best_valid_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss, valid_accuracy = validate(model, valid_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}")
        writer.add_scalar('Train Accuracy', train_accuracy, epoch)
        writer.add_scalar('Valid Accuracy', valid_accuracy, epoch)
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            torch.save(model, save_as_model)
            torch.save(model.state_dict(), args.save_as + "state-dict.pth")
            print(f"Best Valid Acc:{valid_accuracy}, saving to {save_as_model}")
