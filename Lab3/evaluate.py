import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
import argparse
sys.path.insert(0, '../')
# from src.sample_unet import UNet
# from src.train_res_unet import resnet34_unet
from src.oxford_pet import load_dataset
from src.utils import dice_loss, calculate_accuracy
from torchmetrics.functional import dice_score
# import torchmetrics
import saved_models


def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_dice = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch['image'].shape[0] == 1:
                continue
            images = batch['image'].to(device, dtype=torch.float32)
            masks = batch['mask'].to(device, dtype=torch.float32)
            
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            accuracy = calculate_accuracy(outputs, masks)
            dice = dice_score(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            running_accuracy += accuracy.item() * images.size(0)
            # print(dice.shape)
            running_dice += dice.item() * images.size(0)
            if batch_idx % 10 == 0:
                print(f"Eval : [{batch_idx * len(batch['image'])}/{len(dataloader.dataset)}"
      f" ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = running_accuracy / len(dataloader.dataset)
    epoch_dice = running_accuracy / len(dataloader.dataset)
    return epoch_loss, epoch_accuracy, epoch_dice


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--use_model', help='unet or resnet34_unet')
    parser.add_argument('--pretrained', type=str, help='Path to load pretrained model')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.pretrained).to(device)
    print("Loaded model:", (args.pretrained))
    print(model)
    data_path = "../dataset"
    batch_size = 4
    test_loader = load_dataset(data_path, mode='test', batch_size=batch_size)
    
    loss, acc, dice = evaluate(model, test_loader, device)
    print(f"Loss: {loss:.4f},  Acc: {acc:.4f}, Dice Score: {dice:.4f}")
    