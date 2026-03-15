import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import rasterio
import numpy as np

from torch.cuda.amp import autocast, GradScaler

from models.teacher_segformer import TeacherSegFormer
from models.teacher_mae import MAETeacher
from models.student_segformer import SegFormerStudent

# ==========================
# DATASET
# ==========================

class CanopyDataset(Dataset):

    def __init__(self, root):

        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")

        self.images = []

        for img in sorted(os.listdir(self.image_dir)):

            mask_path = os.path.join(self.mask_dir, img)

            if os.path.exists(mask_path):
                self.images.append(img)

        self.transform = T.ToTensor()
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        with rasterio.open(img_path) as src:
            img = src.read([1,2,3])

        img = np.transpose(img, (1,2,0))
        img = self.transform(img)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        mask = torch.from_numpy(mask).long()

        return {
            "image": img,
            "mask": mask
        }


# ==========================
# TRAINING
# ==========================

def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    dataset = CanopyDataset(args.data_path)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    num_classes = args.num_classes

    print("Loading teachers...")

    teacher_segformer = TeacherSegFormer(
        args.segformer_path,
        num_classes
    ).to(device)

    teacher_mae = MAETeacher(
        args.mae_weights
    ).to(device)

    print("Loading student...")

    student = SegFormerStudent(
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr
    )

    scaler = GradScaler()

    alpha = args.alpha
    beta = args.beta

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):

        student.train()

        total_loss = 0

        for batch in loader:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            # Teachers (no gradients)
            with torch.no_grad():
                with autocast():

                    segformer_teacher_logits = teacher_segformer(images)
                    mae_teacher_logits = teacher_mae(images)

                    mae_teacher_logits = F.interpolate(
                        mae_teacher_logits,
                        size=segformer_teacher_logits.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

            # Student
            with autocast():

                student_logits = student(images)

                loss_teacher = F.mse_loss(
                    student_logits,
                    segformer_teacher_logits
                )

                loss_mae = F.mse_loss(
                    student_logits,
                    mae_teacher_logits
                )

                loss_supervision = F.cross_entropy(
                    student_logits,
                    masks
                )

                loss = alpha * loss_mae + beta * loss_teacher + loss_supervision

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        ckpt = os.path.join(
            args.checkpoint_dir,
            f"student_epoch_{epoch+1}.pth"
        )

        torch.save(student.state_dict(), ckpt)

        print("Checkpoint guardado:", ckpt)


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--mae_weights", type=str, required=True)

    parser.add_argument("--segformer_path", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--beta", type=float, default=0.5)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    train(args)
