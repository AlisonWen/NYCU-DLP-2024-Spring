import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from sagan_models import Generator
import sys
import argparse
sys.path.insert(0, '../file')
# from evaluator import evaluation_model
from evaluator import evaluation_model

# Load the JSON files
with open('../file/objects.json', 'r') as f:
    objects = json.load(f)
with open('../file/test.json', 'r') as f:
    test_data = json.load(f)

# Function to convert object names to one-hot encoding
def get_one_hot_labels(objects, data, num_classes=24):
    labels_one_hot = torch.zeros(len(data), num_classes)
    for i, objs in enumerate(data):
        for obj in objs:
            labels_one_hot[i, objects[obj]] = 1
    return labels_one_hot

# Function to denormalize images
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

if __name__=="__main__":
    # Parameters
    batch_size = 32
    z_dim = 128
    num_classes = 24
    image_size = 64
    conv_dim = 64
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument('--save-name', type=str)
    args = parser.parse_args()
    # Initialize the Generator and load the pre-trained model
    G = Generator(batch_size=batch_size, image_size=image_size, z_dim=z_dim, conv_dim=conv_dim, num_classes=num_classes).cuda()
    G.load_state_dict(torch.load(args.ckpt_path))
    G.eval()
    # 195014_G.pth, 7361, 7619
    # 444542_G
    # Initialize the evaluation model
    evaluator = evaluation_model()

    # Prepare the test data
    labels = get_one_hot_labels(objects, test_data, num_classes).cuda()
    z = torch.randn(len(test_data), z_dim).cuda()

    # Generate images
    with torch.no_grad():
        fake_images, _, _ = G(z, labels)

    # Save generated images
    save_image(denorm(fake_images.data), 'generated_images.png', nrow=8)

    # Evaluate the generated images
    acc = evaluator.eval(fake_images, labels)
    print(f"Test Evaluation Accuracy: {acc:.4f}")

    # Save individual images
    # for i, img in enumerate(fake_images):
    save_image(denorm(fake_images), f'generated_image_{args.save_name}.png')

    '''******************'''

    with open('../file/objects.json', 'r') as f:
        objects = json.load(f)
    with open('../file/new_test.json', 'r') as f:
        test_data = json.load(f)
        
    labels = get_one_hot_labels(objects, test_data, num_classes).cuda()
    z = torch.randn(len(test_data), z_dim).cuda()

    # Generate images
    with torch.no_grad():
        fake_images, _, _ = G(z, labels)

    # Save generated images
    # save_image(denorm(fake_images.data), 'generated_images.png', nrow=8)

    # Evaluate the generated images
    acc = evaluator.eval(fake_images, labels)
    print(f"New Test Evaluation Accuracy: {acc:.4f}")

    # Save individual images
    # for i, img in enumerate(fake_images):
    save_image(denorm(fake_images), f'new_test_generated_image_{args.save_name}.png')
