import torch
import torch.nn as nn


# Define a minimal Vision Transformer
class MiniViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=64, num_heads=4):
        super(MiniViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unfold(2, 4, 4).unfold(3, 4, 4).reshape(batch_size, -1, 48)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        x, _ = self.attention(x, x, x)
        return self.classifier(x[:, 0])


# Example usage
x = torch.randn(2, 3, 32, 32)  # Random 32x32 RGB images
model = MiniViT()
output = model(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
