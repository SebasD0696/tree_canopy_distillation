# ==========================
# IMPORTS
# ==========================
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from models.teacher_segformer import TeacherSegFormer
from models.teacher_mae import MAETeacher
from models.student_segformer import SegFormerStudent


# ==========================
# DATASET MULTI-GSD (3 escalas simultáneas)
# ==========================
class MultiGSDDataset(Dataset):
    """
    Carga los 3 GSDs (0.1m, 0.2m, 0.4m) del mismo tile simultáneamente.

    Estructura esperada:
        data/images/0.1m/tile_001.png
        data/images/0.2m/tile_001.png
        data/images/0.4m/tile_001.png
        data/masks/0.1m/tile_001.png   ← máscara de referencia (mayor resolución)
    """
    def __init__(self, root_dir, gsd_list=("0.1m", "0.2m", "0.4m"), img_size=512, transform=None):
        self.root_dir = root_dir
        self.gsd_list = gsd_list
        self.img_size = img_size
        self.transform = transform

        # Usar el GSD más fino como referencia para los nombres de archivo
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

        # Máscara del GSD más fino (0.1m) como ground truth principal
        mask = self._load_mask(self.gsd_list[0], filename)

        return {"images": images, "mask": mask, "filename": filename}


# ==========================
# FUSIÓN MULTI-ESCALA (FPN-style)
# ==========================
class MultiScaleFusion(nn.Module):
    """
    Fusiona los logits de los 3 GSDs antes de calcular las pérdidas.
    Aprende pesos de atención por canal para ponderar cada escala.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(num_classes * 3, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1),  # 3 pesos, uno por escala
            nn.Softmax(dim=1)
        )

    def forward(self, logits_01, logits_02, logits_04, target_size):
        # Llevar todas las escalas al mismo tamaño
        l01 = F.interpolate(logits_01, size=target_size, mode="bilinear", align_corners=False)
        l02 = F.interpolate(logits_02, size=target_size, mode="bilinear", align_corners=False)
        l04 = F.interpolate(logits_04, size=target_size, mode="bilinear", align_corners=False)

        # Calcular pesos de atención
        concat = torch.cat([l01, l02, l04], dim=1)  # (B, C*3, H, W)
        weights = self.attention(concat)             # (B, 3, H, W)

        w01 = weights[:, 0:1, :, :]
        w02 = weights[:, 1:2, :, :]
        w04 = weights[:, 2:3, :, :]

        fused = w01 * l01 + w02 * l02 + w04 * l04
        return fused


# ==========================
# KL DIVERGENCE LOSS con temperatura
# ==========================
def kl_loss(student_logits, teacher_logits, temperature=4.0):
    """
    Distilación clásica con KL divergence y temperatura T.
    T alto → distribución más suave → el student aprende más de los logits negativos.
    """
    student_log_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft     = F.softmax(teacher_logits  / temperature, dim=1)
    loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
    return loss * (temperature ** 2)


# ==========================
# TRAINING
# ==========================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    gsd_list = ["0.1m", "0.2m", "0.4m"]

    dataset = MultiGSDDataset(
        root_dir=args.data_path,
        gsd_list=gsd_list,
        img_size=args.img_size,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    num_classes = args.num_classes

    # ---------- Modelos ----------
    print("Cargando teachers...")
    teacher_segformer = TeacherSegFormer(args.segformer_path, num_classes).to(device)
    teacher_mae       = MAETeacher(args.mae_weights).to(device)
    teacher_segformer.eval()
    teacher_mae.eval()

    print("Cargando student...")
    student = SegFormerStudent(num_classes=num_classes).to(device)

    fusion = MultiScaleFusion(num_classes=num_classes).to(device)

    # ---------- Optimizer ----------
    # Student + fusión se entrenan juntos
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(fusion.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    scaler = GradScaler()

    alpha = args.alpha   # peso loss MAE teacher
    beta  = args.beta    # peso loss SegFormer teacher
    T_kd  = args.temperature

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        student.train()
        fusion.train()

        total_loss        = 0
        total_loss_mae    = 0
        total_loss_seg    = 0
        total_loss_sup    = 0

        for batch in loader:
            images_01 = batch["images"]["0.1m"].to(device)
            images_02 = batch["images"]["0.2m"].to(device)
            images_04 = batch["images"]["0.4m"].to(device)
            masks     = batch["mask"].to(device)

            optimizer.zero_grad()

            mask_size = (masks.shape[-2], masks.shape[-1])

            # ---------- Teachers (sin gradiente) ----------
            with torch.no_grad():
                with autocast():
                    # SegFormer teacher sobre la escala más fina
                    seg_teacher_01 = teacher_segformer(images_01)
                    seg_teacher_01 = F.interpolate(seg_teacher_01, size=mask_size,
                                                   mode="bilinear", align_corners=False)

                    # MAE teacher sobre las 3 escalas
                    mae_01 = teacher_mae(images_01)
                    mae_02 = teacher_mae(images_02)
                    mae_04 = teacher_mae(images_04)

                    mae_01 = F.interpolate(mae_01, size=mask_size, mode="bilinear", align_corners=False)
                    mae_02 = F.interpolate(mae_02, size=mask_size, mode="bilinear", align_corners=False)
                    mae_04 = F.interpolate(mae_04, size=mask_size, mode="bilinear", align_corners=False)

                    # Fusión de MAE teachers → referencia multi-escala
                    mae_teacher_fused = (mae_01 + mae_02 + mae_04) / 3.0

            # ---------- Student sobre las 3 escalas ----------
            with autocast():
                student_01 = student(images_01)
                student_02 = student(images_02)
                student_04 = student(images_04)

                # Fusión multi-escala con atención aprendida
                student_fused = fusion(student_01, student_02, student_04, mask_size)

                # ---------- Pérdidas ----------
                # 1. KL vs MAE teacher (multi-escala)
                loss_mae = kl_loss(student_fused, mae_teacher_fused, temperature=T_kd)

                # 2. KL vs SegFormer teacher (escala fina)
                loss_seg = kl_loss(
                    F.interpolate(student_01, size=mask_size, mode="bilinear", align_corners=False),
                    seg_teacher_01,
                    temperature=T_kd
                )

                # 3. Supervisión con ground truth
                loss_sup = F.cross_entropy(student_fused, masks)

                loss = alpha * loss_mae + beta * loss_seg + loss_sup

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss     += loss.item()
            total_loss_mae += loss_mae.item()
            total_loss_seg += loss_seg.item()
            total_loss_sup += loss_sup.item()

        scheduler.step()

        n = len(loader)
        avg_loss     = total_loss     / n
        avg_loss_mae = total_loss_mae / n
        avg_loss_seg = total_loss_seg / n
        avg_loss_sup = total_loss_sup / n

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss_mae: {avg_loss_mae:.4f} | "
            f"loss_seg: {avg_loss_seg:.4f} | "
            f"loss_sup: {avg_loss_sup:.4f} | "
            f"total: {avg_loss:.4f}"
        )

        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(args.checkpoint_dir, "student_best.pth")
            torch.save({
                "epoch": epoch + 1,
                "student": student.state_dict(),
                "fusion": fusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": best_loss
            }, best_ckpt)
            print(f"  → Mejor modelo guardado (loss: {best_loss:.4f})")

        # Checkpoint periódico
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            ckpt = os.path.join(args.checkpoint_dir, f"student_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "student": student.state_dict(),
                "fusion": fusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss
            }, ckpt)
            print(f"  → Checkpoint guardado: {ckpt}")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",       type=str,   required=True,
                        help="Ruta raíz del dataset (contiene images/ y masks/)")
    parser.add_argument("--mae_weights",     type=str,   required=True,
                        help="Pesos del MAE Teacher")
    parser.add_argument("--segformer_path",  type=str,   required=True,
                        help="Ruta al SegFormer Teacher")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--img_size",        type=int,   default=512,
                        help="Tamaño al que se redimensionan todas las imágenes")
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--num_classes",     type=int,   default=2)
    parser.add_argument("--alpha",           type=float, default=0.4,
                        help="Peso de la loss MAE teacher")
    parser.add_argument("--beta",            type=float, default=0.4,
                        help="Peso de la loss SegFormer teacher")
    parser.add_argument("--temperature",     type=float, default=4.0,
                        help="Temperatura para KL divergence distillation")
    parser.add_argument("--checkpoint_dir",  type=str,   default="checkpoints")
    args = parser.parse_args()
    train(args)
