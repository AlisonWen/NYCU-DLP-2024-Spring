import torch
import torchvision.datasets as dsets
from torchvision import transforms
import json
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset


# class Data_Loader():
#     def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
#         self.dataset = dataset
#         self.path = image_path
#         self.imsize = image_size
#         self.batch = batch_size
#         self.shuf = shuf
#         self.train = train

#     def transform(self, resize, totensor, normalize, centercrop):
#         options = []
#         if centercrop:
#             options.append(transforms.CenterCrop(160))
#         if resize:
#             options.append(transforms.Resize((self.imsize,self.imsize)))
#         if totensor:
#             options.append(transforms.ToTensor())
#         if normalize:
#             options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#         transform = transforms.Compose(options)
#         return transform

#     def load_lsun(self, classes='church_outdoor_train'):
#         transforms = self.transform(True, True, True, False)
#         dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
#         return dataset

#     def load_celeb(self):
#         transforms = self.transform(True, True, True, True)
#         dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
#         return dataset


#     def loader(self):
#         if self.dataset == 'lsun':
#             dataset = self.load_lsun()
#         elif self.dataset == 'celeb':
#             dataset = self.load_celeb()

#         loader = torch.utils.data.DataLoader(dataset=dataset,
#                                               batch_size=self.batch,
#                                               shuffle=self.shuf,
#                                               num_workers=2,
#                                               drop_last=True)
#         return loader

class iCLEVRDataset(Dataset):
    def __init__(self, data_file, objects_file, root_dir):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        with open(objects_file, 'r') as f:
            self.objects = json.load(f)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = list(self.data.keys())[idx]

        img_name = os.path.join(self.root_dir, idx)
        image = Image.open(img_name).convert('RGB')

        labels = self.data[idx]
        labels_idx = [self.objects[obj] for obj in labels]
        labels_one_hot = torch.zeros(len(self.objects))
        labels_one_hot[labels_idx] = 1

        if self.transform:
            image = self.transform(image)
        # print(f"in iCLEVRDataset: {image.shape}")
        return image, labels_one_hot

class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, data_file='../file/train.json', objects_file='../file/objects.json', shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train
        self.data_file = data_file
        self.objects_file = objects_file

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        # if centercrop:
        #     options.append(transforms.CenterCrop(160))
        # if resize:
        #     options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transform = self.transform(False, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transform)
        return dataset

    def load_celeb(self):
        transform = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transform)
        return dataset

    def load_iclevr(self):
        dataset = iCLEVRDataset(data_file=self.data_file, objects_file=self.objects_file, root_dir='../iclevr')
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'iclevr':
            dataset = self.load_iclevr()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=self.shuf,
                                             num_workers=2,
                                             drop_last=True)
        return loader