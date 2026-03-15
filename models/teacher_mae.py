import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mae_encoder import build_mae_encoder

class MAETeacher(nn.Module):
    def __init__(self, encoder, weights_path, num_classes=2):
        super().__init__()
        self.encoder = encoder

        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']

        # Interpolar pos_embed aunque el tamaño de entrada sea 128
        if 'pos_embed' in state_dict:
            pos_embed_checkpoint = state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            cls_token = pos_embed_checkpoint[:, 0:1, :]
            patch_pos_embed = pos_embed_checkpoint[:, 1:, :]

            num_patches_new = (encoder.img_size // encoder.patch_size) ** 2
            num_patches_old = patch_pos_embed.shape[1]

            if num_patches_old != num_patches_new:
                size_old = int(num_patches_old ** 0.5)
                size_new = int(num_patches_new ** 0.5)
                patch_pos_embed = patch_pos_embed.reshape(1, size_old, size_old, embedding_size).permute(0,3,1,2)
                patch_pos_embed = F.interpolate(patch_pos_embed, size=(size_new, size_new), mode='bicubic', align_corners=False)
                patch_pos_embed = patch_pos_embed.permute(0,2,3,1).reshape(1, num_patches_new, embedding_size)
                state_dict['pos_embed'] = torch.cat([cls_token, patch_pos_embed], dim=1)

        self.encoder.load_state_dict(state_dict, strict=False)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        """
        x: [B, C, H, W], puede ser 512x512
        Se hace sliding window 128x128 y se recombina.
        """
        B, C, H, W = x.shape
        crop_size = 128
        stride = 128  # no overlap por simplicidad

        # Crear salida vacía
        out = torch.zeros(B, self.decoder[-1].out_channels, H, W, device=x.device)

        # Recorrer crops
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                crop = x[:, :, i:i+crop_size, j:j+crop_size]
                logits_crop = self.forward_crop(crop)
                out[:, :, i:i+crop_size, j:j+crop_size] = logits_crop

        return out

    def forward_crop(self, x_crop):
        """
        Forward de un crop 128x128
        """
        latent, _, _ = self.encoder.forward_encoder(x_crop, mask_ratio=0)
        tokens = latent[:, 1:, :]
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        features = tokens.transpose(1, 2).reshape(B, C, H, W)
        logits = self.decoder(features)

        # Upsample a tamaño de crop original
        logits = F.interpolate(logits, size=(x_crop.shape[2], x_crop.shape[3]),
                            mode='bilinear', align_corners=False)
        return logits