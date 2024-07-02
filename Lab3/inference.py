import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
import argparse
sys.path.insert(0, '../')
from src.models.unet import UNet
from src.models.resnet34_unet import resnet34_unet
from src.oxford_pet import load_dataset
from src.utils import dice_loss, calculate_accuracy
from src.evaluate import evaluate

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--use_model', help='unet or resnet34_unet')
    parser.add_argument('--pretrained', type=str, help='Path to load pretrained model')
    parser.add_argument('--model', help='unet or res')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=1)
    else:
        model = resnet34_unet(num_classes=1)
    model.load_state_dict(torch.load(args.pretrained))
    # model = torch.load(args.pretrained).to(device)
    model = model.to(device)
    print("Loaded model:", (args.pretrained))
    print(model)
    data_path = "../dataset"
    batch_size = 4
    test_loader = load_dataset(data_path, mode='test', batch_size=batch_size)
    
    loss, acc, dice = evaluate(model, test_loader, device)
    print(f"Loss: {loss:.4f},  Acc: {acc:.4f}, Dice Score: {dice:.4f}")