import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from sagan_models import Generator, Discriminator
from data_loader import iCLEVRDataset
from torch.utils.data import DataLoader
import json
from utils import *
import wandb
import sys
sys.path.insert(0, '../file')
from evaluator import evaluation_model

class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()
        
        self.eval_model = evaluation_model()
        
        self.test_dataset = iCLEVRDataset(data_file='../file/test.json', objects_file='../file/objects.json', root_dir='../iclevr')
        self.test_loader = DataLoader(self.test_dataset, batch_size=10, shuffle=False)
        
        with open('../file/test.json', 'r') as f:
            test_data = json.load(f)

        # Load object labels
        with open('../file/objects.json', 'r') as f:
            objects = json.load(f)
        test_labels = []
        for labels in test_data:
            labels_idx = [objects[obj] for obj in labels]
            labels_one_hot = torch.zeros(len(objects))
            labels_one_hot[labels_idx] = 1
            test_labels.append(labels_one_hot)
        self.test_labels = torch.stack(test_labels).to("cuda")

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

        # Initialize wandb
        run = wandb.init(project="DLP_Lab6",
                     config=vars(config),
                     sync_tensorboard=True,
                     name=config.wandb_run_name,
                     save_code=True)
        config.tensorboard_log = f"runs/{config.wandb_run_name}"
        wandb.watch(self.G, log="all")
        wandb.watch(self.D, log="all")

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
        fixed_labels = tensor2var(torch.zeros(self.batch_size, self.data_loader.dataset[0][1].shape[0]))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:
                real_images, labels = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, labels = next(data_iter)

            # Compute loss with real images
            # dr1, dr2, df1, gf1, gf2 are attention scores
            real_images = tensor2var(real_images)
            # print(f"real_images: {real_images.shape}")
            labels = tensor2var(labels)
            d_out_real, dr1, dr2 = self.D(real_images, labels)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images, gf1, gf2 = self.G(z, labels)
            d_out_fake, df1, df2 = self.D(fake_images, labels)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out, _, _ = self.D(interpolated, labels)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images, _, _ = self.G(z, labels)

            # Compute loss with fake images
            g_out_fake, _, _ = self.D(fake_images, labels)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            # Evaluate the model
            with torch.no_grad():
                eval_acc = self.eval_model.eval(fake_images, labels)

            # Log metrics to wandb
            wandb.log({
                "d_loss_real": d_loss_real.item(),
                "d_loss_fake": d_loss_fake.item(),
                "g_loss_fake": g_loss_fake.item(),
                "eval_accuracy": eval_acc,
                "step": step + 1
            })

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}, Eval Acc: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step, d_loss_real.item(),
                             self.G.attn1.gamma.mean().item(), self.G.attn2.gamma.mean().item(), eval_acc))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images, _, _ = self.G(fixed_z, fixed_labels)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))
                wandb.log({"generated_images": [wandb.Image(fake_images.data, caption=f"Step {step + 1}")]})

            if (step + 1) % model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
                wandb.save(os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                wandb.save(os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
                
         

    def build_model(self):
        self.G = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim, num_classes=24).cuda()
        self.D = Discriminator(self.batch_size, self.imsize, self.d_conv_dim, num_classes=24).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
