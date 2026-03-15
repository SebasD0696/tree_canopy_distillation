# training/train_distillation.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import rasterio
from torchvision import transforms as T

from models.teacher_segformer import TeacherSegFormer
from models.teacher_mae import MAETeacher
from models.student_segformer import StudentSegFormer
from models.mae_encoder import build_mae_encoder

# -------------------------------
# Configuración
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
batch_size = 2
lr = 1e-4
num_epochs = 10
alpha = 0.5  # peso MAE teacher
beta = 0.5   # peso SegFormer teacher
checkpoint_dir = "outputs/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# -------------------------------
# 1️⃣ Inicializar modelos
# -------------------------------
teacher_segformer = TeacherSegFormer("models/teacher_segformer", num_classes).to(device)
teacher_mae = MAETeacher(build_mae_encoder(), "cross_scale_mae_large_pretrain.pth", num_classes).to(device)

student = StudentSegFormer("restor/tcd-segformer-mit-b0", num_classes).to(device)

teacher_segformer.eval()
teacher_mae.eval()
for p in teacher_segformer.parameters():
    p.requires_grad = False
for p in teacher_mae.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(student.parameters(), lr=lr)

# -------------------------------
# 2️⃣ Dataset para tree canopy
# -------------------------------
class TreeCanopyDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
        self.masks_files = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):
        # Leer imagen RGB
        with rasterio.open(self.images_files[idx]) as src:
            img = src.read([1, 2, 3]).astype("float32") / 255.0
            img = torch.from_numpy(img)
        
        # Leer máscara
        with rasterio.open(self.masks_files[idx]) as src:
            mask = src.read(1).astype("int64")
            mask = torch.from_numpy(mask)

        if self.transform:
            img = self.transform(img)

        return {"image": img, "mask": mask}

dataset = TreeCanopyDataset(images_dir="data/images", masks_dir="data/masks", transform=None)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# 3️⃣ Entrenamiento con distillation
# -------------------------------
for epoch in range(num_epochs):
    student.train()
    total_loss = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.no_grad():
            segformer_teacher_logits = teacher_segformer(images)
            mae_teacher_logits = teacher_mae(images)

        # Forward student
        student_logits = student(images)

        # Distillation losses
        loss_teacher = F.mse_loss(student_logits, segformer_teacher_logits)
        loss_mae = F.mse_loss(student_logits, mae_teacher_logits)
        loss_supervision = F.cross_entropy(student_logits, masks)

        loss = alpha * loss_mae + beta * loss_teacher + loss_supervision

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # Guardar checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"student_epoch_{epoch+1}.pth")
    torch.save(student.state_dict(), ckpt_path)
    print(f"Checkpoint guardado: {ckpt_path}")