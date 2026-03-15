import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class SegFormerStudent(nn.Module):
    """
    Student SegFormer B0 que aprende mediante distillation
    de dos teachers:
        - Teacher SegFormer B5 (segmentación)
        - Teacher MAE (features multi-resolution)
    """
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", num_classes=2):
        super().__init__()
        
        # Modelo SegFormer Student
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes
        )

    def forward(self, x):
        """
        Forward normal del student
        """
        outputs = self.model(pixel_values=x)
        return outputs.logits

    def forward_with_teachers(self, x, teacher_segformer_logits=None, teacher_mae_embeddings=None, alpha=0.5, beta=0.5):
        """
        Forward con distillation de ambos teachers

        Args:
            x: imágenes de entrada [B, C, H, W]
            teacher_segformer_logits: logits del Teacher SegFormer
            teacher_mae_embeddings: embeddings del Teacher MAE
            alpha: peso loss MAE
            beta: peso loss SegFormer
        """
        student_outputs = self.model(pixel_values=x)
        student_logits = student_outputs.logits

        loss = None

        # Distillation loss
        if teacher_segformer_logits is not None or teacher_mae_embeddings is not None:
            loss = 0.0

            # 1️⃣ Distillation de logits del Teacher SegFormer
            if teacher_segformer_logits is not None:
                # MSE entre logits
                loss_segformer = F.mse_loss(student_logits, teacher_segformer_logits)
                loss += beta * loss_segformer

            # 2️⃣ Distillation de features del Teacher MAE
            if teacher_mae_embeddings is not None:
                # Redimensionar student_logits a embeddings del MAE si es necesario
                student_feats = F.interpolate(student_logits, size=teacher_mae_embeddings.shape[-2:], mode='bilinear', align_corners=False)
                loss_mae = F.mse_loss(student_feats, teacher_mae_embeddings)
                loss += alpha * loss_mae

        return student_logits, loss