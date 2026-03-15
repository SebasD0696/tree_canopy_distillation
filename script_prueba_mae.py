import torch
from models.mae_encoder import build_mae_encoder
from models.teacher_mae import MAETeacher

encoder = build_mae_encoder()

model = MAETeacher(
    encoder,
    "weights/cross_scale_mae_large_pretrain.pth",
    num_classes=2
)

# Dummy 512x512
dummy = torch.randn(1,3,512,512)

out = model(dummy)
print("Output shape:", out.shape)  # debería ser [1, 2, 512, 512]