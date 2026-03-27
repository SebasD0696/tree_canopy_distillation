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

    def __init__(self, root_dir, gsd_list=("0.1m", "0.2m", "0.4m"), img_size=512, transform=None):
        self.root_dir = root_dir
        self.gsd_list = gsd_list
        self.img_size = img_size
        self.transform = transform

        ref_gsd = gsd_list[0]
        img_dir = os.path.join(root_dir, "images", ref_gsd)

        self.filenames = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith((".png", ".jpg", ".tif"))
        ])

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, gsd, filename):

        path = os.path.join(self.root_dir, "images", gsd, filename)

        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        if self.transform:
            img = self.transform(img)

        return img

    def _load_mask(self, gsd, filename):

        path = os.path.join(self.root_dir, "masks", gsd, filename)

        mask = Image.open(path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        return torch.from_numpy(np.array(mask)).long()

    def __getitem__(self, idx):

        filename = self.filenames[idx]

        images = {
            gsd: self._load_image(gsd, filename)
            for gsd in self.gsd_list
        }

        mask = self._load_mask(self.gsd_list[0], filename)

        return {"images": images, "mask": mask, "filename": filename}


# ==========================
# FUSIÓN MULTI-ESCALA
# ==========================
class MultiScaleFusion(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(num_classes * 3, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, logits_01, logits_02, logits_04, target_size):

        l01 = F.interpolate(logits_01, size=target_size, mode="bilinear", align_corners=False)
        l02 = F.interpolate(logits_02, size=target_size, mode="bilinear", align_corners=False)
        l04 = F.interpolate(logits_04, size=target_size, mode="bilinear", align_corners=False)

        concat = torch.cat([l01, l02, l04], dim=1)

        weights = self.attention(concat)

        w01 = weights[:, 0:1]
        w02 = weights[:, 1:2]
        w04 = weights[:, 2:3]

        fused = w01 * l01 + w02 * l02 + w04 * l04

        return fused


# ==========================
# KL DISTILLATION LOSS
# ==========================
def kl_loss(student_logits, teacher_logits, temperature=4.0):

    student_log_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

    loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")

    return loss * (temperature ** 2)


# ==========================
# TRAIN
# ==========================
def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    gsd_list = ["0.1m", "0.2m", "0.4m"]

    dataset = MultiGSDDataset(
        root_dir=args.data_path,
        gsd_list=gsd_list,
        img_size=args.img_size,
        transform=transform
    )

    # ==========================
    # SPLIT TRAIN / VAL
    # ==========================
    total_size = len(dataset)

    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    num_classes = args.num_classes

    print("Cargando teachers...")

    teacher_segformer = TeacherSegFormer(args.segformer_path, num_classes).to(device)
    teacher_mae = MAETeacher(args.mae_weights).to(device)

    teacher_segformer.eval()
    teacher_mae.eval()

    print("Cargando student...")

    student = SegFormerStudent(num_classes=num_classes).to(device)

    fusion = MultiScaleFusion(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(fusion.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    scaler = GradScaler()

    alpha = args.alpha
    beta = args.beta
    T_kd = args.temperature

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")

    # ==========================
    # TRAIN LOOP
    # ==========================
    for epoch in range(args.epochs):

        student.train()
        fusion.train()

        total_loss = 0

        for batch in train_loader:

            images_01 = batch["images"]["0.1m"].to(device)
            images_02 = batch["images"]["0.2m"].to(device)
            images_04 = batch["images"]["0.4m"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            mask_size = (masks.shape[-2], masks.shape[-1])

            with torch.no_grad():

                seg_teacher_01 = teacher_segformer(images_01)
                seg_teacher_01 = F.interpolate(seg_teacher_01, size=mask_size, mode="bilinear")

                mae_01 = teacher_mae(images_01)
                mae_02 = teacher_mae(images_02)
                mae_04 = teacher_mae(images_04)

                mae_01 = F.interpolate(mae_01, size=mask_size, mode="bilinear")
                mae_02 = F.interpolate(mae_02, size=mask_size, mode="bilinear")
                mae_04 = F.interpolate(mae_04, size=mask_size, mode="bilinear")

                mae_teacher_fused = (mae_01 + mae_02 + mae_04) / 3

            with autocast():

                student_01 = student(images_01)
                student_02 = student(images_02)
                student_04 = student(images_04)

                student_fused = fusion(student_01, student_02, student_04, mask_size)

                loss_mae = kl_loss(student_fused, mae_teacher_fused, T_kd)

                loss_seg = kl_loss(
                    F.interpolate(student_01, size=mask_size),
                    seg_teacher_01,
                    T_kd
                )

                loss_sup = F.cross_entropy(student_fused, masks)

                loss = alpha * loss_mae + beta * loss_seg + loss_sup

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)

        # ==========================
        # VALIDATION
        # ==========================
        student.eval()
        fusion.eval()

        val_loss = 0

        with torch.no_grad():

            for batch in val_loader:

                images_01 = batch["images"]["0.1m"].to(device)
                images_02 = batch["images"]["0.2m"].to(device)
                images_04 = batch["images"]["0.4m"].to(device)
                masks = batch["mask"].to(device)

                mask_size = (masks.shape[-2], masks.shape[-1])

                student_01 = student(images_01)
                student_02 = student(images_02)
                student_04 = student(images_04)

                student_fused = fusion(student_01, student_02, student_04, mask_size)

                loss = F.cross_entropy(student_fused, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save({
                "epoch": epoch + 1,
                "student": student.state_dict(),
                "fusion": fusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": best_val_loss
            },
                os.path.join(args.checkpoint_dir, "student_best.pth")
            )

            print("→ Mejor modelo guardado")

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mae_weights", type=str, required=True)
    parser.add_argument("--segformer_path", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=512)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.4)

    parser.add_argument("--temperature", type=float, default=4.0)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    train(args)