import torch
from models.teacher_segformer import TeacherSegFormer

model = TeacherSegFormer(
    model_path="weights/segformer_b5",
    num_classes=2
)

dummy = torch.randn(1,3,512,512)

out = model(dummy)

print(out.shape)