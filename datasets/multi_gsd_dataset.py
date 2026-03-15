import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MultiGSDDataset(Dataset):
    def __init__(self, root_dir="data", gsd_list=["0.1m", "0.2m", "0.4m"], transform=None):
        self.samples = []
        self.transform = transform

        for gsd in gsd_list:
            img_dir = os.path.join(root_dir, "images", gsd)
            mask_dir = os.path.join(root_dir, "masks", gsd)
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, img_name)
                if os.path.exists(mask_path):
                    self.samples.append({
                        "img": img_path,
                        "mask": mask_path,
                        "gsd": gsd
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["img"]).convert("RGB")
        mask = Image.open(sample["mask"])
    
        if self.transform:
            image = self.transform(image)
        
        # Convertir máscara a tensor
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
    
        #  Normalizar máscara para que los valores estén en [0, num_classes-1]
        mask = (mask > 0).long()  # todos los valores >0 pasan a 1
    
        return {
            "image": image,
            "mask": mask,
            "gsd": sample["gsd"]
        }
