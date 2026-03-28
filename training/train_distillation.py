# ==========================
# IMPORTS
# ==========================
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from models.teacher_segformer import TeacherSegFormer
from models.teacher_mae import MAETeacher
from models.student_segformer import SegFormerStudent


# ==========================
# DATASET MULTI-GSD
# ==========================
class MultiGSDDataset(Dataset):
    def __init__(self, root_dir, gsd="0.1m", img_size=512, transform=None):
        self.root_dir = root_dir
        self.gsd = gsd
        self.img_size = img_size
        self.transform = transform

        img_dir = os.path.join(root_dir, "images", gsd)
        self.filenames = sorted([f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".tif"))])

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, filename):
        path = os.path.join(self.root_dir, "images", self.gsd, filename)
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return img

    def _load_mask(self, filename):
        path = os.path.join(self.root_dir, "masks", self.gsd, filename)
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.int64)
        return torch.from_numpy(mask).long()

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = self._load_image(filename)
        mask = self._load_mask(filename)
        return {"image": image, "mask": mask, "filename": filename}


# ==========================
# KL LOSS
# ==========================
def kl_loss(student_logits, teacher_logits, T=4):
    student_log = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    loss = F.kl_div(student_log, teacher_soft, reduction="batchmean")
    return loss * (T * T)


# ==========================
# TRAIN
# ==========================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.backends.cudnn.benchmark = True

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    dataset = MultiGSDDataset(args.data_path, img_size=args.img_size, transform=transform)

    # ======================
    # SPLIT TRAIN / VAL
    # ======================
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("Train:", len(train_dataset), "Val:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = args.num_classes

    # ======================
    # LOAD TEACHERS
    # ======================
    teacher_segformer = TeacherSegFormer(args.segformer_path, num_classes).to(device)
    teacher_mae = MAETeacher(args.mae_weights).to(device)
    teacher_segformer.eval()
    teacher_mae.eval()
    for p in teacher_segformer.parameters():
        p.requires_grad = False
    for p in teacher_mae.parameters():
        p.requires_grad = False

    # ======================
    # LOAD STUDENT
    # ======================
    student = SegFormerStudent(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    best_loss = float("inf")

    # ======================
    # TRAIN LOOP
    # ======================
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            mask_size = masks.shape[-2:]

            optimizer.zero_grad()

            # ======================
            # TEACHER FEATURES MULTI-SCALE
            # ======================
            with torch.no_grad():
                # Teacher logits
                teacher_logits = teacher_segformer(images)
                teacher_logits = F.interpolate(teacher_logits, size=mask_size)

                # MAE features multi-layer (simula multi-scale)
                mae_feats = teacher_mae(images)
                mae_feats = F.interpolate(mae_feats, size=mask_size)

            # ======================
            # STUDENT
            # ======================
            with autocast():
                student_logits = student(images)

                # Loss: cross-entropy supervisada
                loss_sup = F.cross_entropy(student_logits, masks)

                # Distillation de logits
                loss_kl = kl_loss(student_logits, teacher_logits, args.temperature)

                # Distillation de features MAE
                loss_mae = F.mse_loss(student_logits, mae_feats)

                # Loss total
                loss = args.alpha * loss_mae + args.beta * loss_kl + loss_sup

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # ======================
        # VALIDATION
        # ======================
        student.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                mask_size = masks.shape[-2:]

                pred = student(images)
                loss = F.cross_entropy(pred, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} TrainLoss {train_loss:.4f} ValLoss {val_loss:.4f}")

        # Guardar mejor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "student": student.state_dict(),
                "epoch": epoch
            }, os.path.join(args.checkpoint_dir, "student_best.pth"))
            print("Best model saved")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mae_weights", type=str, required=True)
    parser.add_argument("--segformer_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)
