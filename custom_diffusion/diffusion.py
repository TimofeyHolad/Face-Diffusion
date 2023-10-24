import math
from tqdm import trange
import torch
import torch.nn as nn
import torchvision.transforms as tt
import matplotlib.pyplot as plt


class Diffusion:
    def __init__(self, Unet, T=1000, beta_start=1e-4, beta_end=0.02, img_size=64, model_size=5, base_channels=64, lr=1e-4, optimizer=torch.optim.AdamW, criterion=nn.MSELoss, device='cpu', dtype=torch.float16):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.model_size = model_size
        self.lr = lr
        self.device = device
        self.dtype = dtype
        
        self.betas = self.cos_betas_scheduler().to(device, dtype=dtype)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.Unet = Unet(img_size=img_size, model_size=model_size, timestamps_num=T, in_channels=3, base_channels=64, out_channels=3).to(device, dtype=dtype)
        self.optimizer = optimizer(self.Unet.parameters(), lr=lr)
        self.criterion = criterion()
        self.scaler = torch.cuda.amp.GradScaler()
        
    @torch.no_grad()
    def linear_betas_scheduler(self):
        return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.T)
    
    @torch.no_grad()
    def cos_encoder(self, t, s=8e-3):
        return torch.cos((math.pi / 2) * ((t + 1) / self.T + s) / (1 + s)) ** 2
    
    @torch.no_grad()
    def cos_betas_scheduler(self):
        start_alpha = self.cos_encoder(torch.tensor(-1))
        alphas_cumprod = self.cos_encoder(torch.arange(start=-1, end=self.T))
        alphas_cumprod /= start_alpha
        betas = [1 - alphas_cumprod[t+1] / alphas_cumprod[t] for t in range(self.T)]
        return torch.stack(betas)
        
    @torch.no_grad()
    def get_noisy_image(self, x, t):
        if isinstance(t, int):
            t = torch.tensor(t)[None]
        noise = torch.randn_like(x, device=self.device, dtype=self.dtype)
        return x * torch.sqrt(self.alphas_cumprod[t])[:, None, None, None] + noise * torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None], noise
    
    def save_state_dict(self, checkpoint_path):
        torch.save({
            'Unet': self.Unet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }, checkpoint_path)
        
    def load_state_dict(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.Unet.load_state_dict(checkpoint['Unet'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.Unet.to(self.device)
    
    def train_backward(self, loader, epochs=50, checkpoint_path='checkpoint.pth'):
        control_noise = torch.randn(size=(1, 3, self.img_size, self.img_size), device=self.device, dtype=self.dtype)
        losses = []
        
        self.Unet.train()
        for epoch in trange(epochs):
            print('---------------------------------------------')
            print(f'{epoch+1}/{epochs}:')
            epoch_loss = 0
            for x in loader:
                self.optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    t = torch.randint(high=self.T, size=(x.shape[0],))
                    noisy_x, noise = self.get_noisy_image(x, t)
                    noise_pred = self.Unet(noisy_x, t)
                    loss = self.criterion(noise_pred, noise)
                    #loss.backward()
                #optimizer.step()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_loss += float(loss)
            print(f'loss:{epoch_loss}')
            losses.append(epoch_loss)
            if (epoch + 1) % 10 == 0:
                self.backward_show(control_noise)
            if (epoch + 1) % 50 == 0:
                self.save_state_dict(checkpoint_path=checkpoint_path)
            #img = self.sample(control_noise)
            #self.image_show(img)
            plt.show()
        return losses
    
    @torch.no_grad()
    def sample(self, x=None):
        if x is None:
            x = torch.randn(size=(1, 3, self.img_size, self.img_size), device=self.device, dtype=self.dtype)
        self.Unet.eval()
        for t in reversed(range(self.T)):
            yield x[0]
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                noise = self.Unet(x, t)
            x = (x - ((1 - self.alphas[t])/torch.sqrt(1 - self.alphas_cumprod[t])) * noise)
            z = torch.randn_like(x)
            sigma = torch.sqrt(self.betas[t] * (1 - self.alphas_cumprod[t-1]) / (1 - self.alphas_cumprod[t]))
            x += z * sigma
        x -= z * sigma
        yield x[0]
    
    @torch.no_grad()
    def image_show(self, x):
        if len(x.shape) == 4:
            x = x[0]
        transforms = tt.Compose([
            tt.Lambda(lambda t: (t + 1) / 2),
            tt.Lambda(lambda t: torch.permute(t, dims=(1, 2, 0))),
            tt.Lambda(lambda t: t.cpu()),
        ])
        x = transforms(x).to(dtype=torch.float32)
        plt.imshow(x)
        plt.axis('off')
    
    @torch.no_grad()
    def forward_show(self, x):
        plt.figure(figsize=(18, 6))
        ncols = self.T // 100 + 1
        for t in range(-1, self.T, 100):
            index = (t + 1) // 100
            plt.subplot(1, ncols, index + 1)
            if index == 0:
                img = x
            else:
                img = self.get_noisy_image(x, t)[0]
            self.image_show(img)
            
    @torch.no_grad()
    def backward_show(self, x=None):
        if x is None:
            x = torch.randn(size=(1, 3, self.img_size, self.img_size), device=self.device, dtype=self.dtype)
        plt.figure(figsize=(18, 6))
        ncols = self.T // 100 + 1
        for t, x in enumerate(self.sample(x)):
            if t % 100 == 0:
                index = t // 100
                plt.subplot(1, ncols, index + 1)
                self.image_show(x)
        plt.subplot(1, ncols, ncols)
        self.image_show(x)