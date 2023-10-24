from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tt
from PIL import Image


class FaceDataset(Dataset):
    
    def load_img(self, file):
        image = Image.open(file)
        image.load()
        return image
    
    def __init__(self, path, dataset_size=3143, img_size=64, device='cpu', dtype=torch.float32):
        super().__init__()
        assert dataset_size <= 3143, 'dataset_size is too big'
        path = Path(path)
        self.files = sorted(list(path.rglob('*.png')))[:dataset_size]
        self.size = (img_size, img_size)
        self.len_ = len(self.files)
        transforms = tt.Compose([
            tt.Resize(size=self.size),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Lambda(lambda t: (t * 2) - 1)
        ])
        imgs = [transforms(self.load_img(file)) for file in self.files]
        self.imgs = torch.stack(imgs).to(device, dtype=dtype)
        
    def __len__(self):
        return self.len_
    
    def __getitem__(self, index):
        return self.imgs[index]