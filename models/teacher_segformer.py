import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class TeacherSegFormer(nn.Module):

    def __init__(self, model_path, num_classes):

        super().__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_path,
            num_labels=num_classes
        )

        # congelar pesos (teacher no se entrena)
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    def forward(self, x):

        outputs = self.model(pixel_values=x)

        return outputs.logits