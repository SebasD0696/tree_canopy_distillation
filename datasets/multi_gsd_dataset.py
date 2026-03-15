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
    
        # Transform imagen
        image = image.resize((512, 512))  #  resize de la imagen
        image = transforms.ToTensor()(image)
        
        mask = mask.resize((512, 512))  # resize de la máscara
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask = (mask > 0).long()
        return {
            "image": image,
            "mask": mask,
            "gsd": sample["gsd"]
        }
