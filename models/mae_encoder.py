import torch
from cross_scale_mae.models_mae.MAE_ViT_Baseline import MAE_ViT_Baseline

def build_mae_encoder():
    """
    Construye el encoder MAE Large para imágenes 128x128.
    """
    model = MAE_ViT_Baseline(
        img_size=128,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,  # Large
        depth=24,
        num_heads=16
    )
    return model