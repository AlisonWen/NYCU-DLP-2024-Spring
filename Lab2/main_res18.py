import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, './')
from dataloader import getData, BufferflyMothLoader
from ResNet50 import RN50
import argparse

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f'Epoch {epoch} complete! Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    
    return accuracy
    
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)

def main():
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', help="Tensorboard log path name", type=str, default=None)
    parser.add_argument('--model_name', help="The name of the saved model", type=str, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using', device, torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    # a = 1/0
    if args.log_path is not None:
        writer = SummaryWriter(args.log_path)
    else:
        writer = SummaryWriter('./Res18-3')
        
    root_dir = './dataset/' 
    batch_size = 32
    epochs = 80
    
    # Data loading
    train_dataset = BufferflyMothLoader(root=root_dir, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = BufferflyMothLoader(root=root_dir, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = BufferflyMothLoader(root=root_dir, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    model = RN50().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model_name = "model-80-epoch-real-res18-3.pth"
    if args.model_name is not None:  # Use 'args' to access the 'model_name'
        model_name = args.model_name + '.pth'  # Fix typo here from 'model-name' to 'model_name'
    acc_best = 0.0
    
    for epoch in range(1, epochs + 1):
        train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
        valid_accuracy = test(model, device, val_loader, criterion)
        test_accuracy = test(model, device, test_loader, criterion)
        writer.add_scalar('Training Accuracy', train_accuracy, epoch)
        writer.add_scalar('Validation Accuracy', valid_accuracy, epoch)
        writer.add_scalar('Test Accuracy', test_accuracy, epoch)
        if valid_accuracy > acc_best:
            acc_best = valid_accuracy
            print("model saved")
            torch.save(model, model_name)

if __name__ == '__main__':
    main()
