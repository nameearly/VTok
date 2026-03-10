import torch
import torch.nn as nn

class MotionEncoder(nn.Module):
    """
    Computes single motion token from the residual between a frame's feature and the key frame's features.
    """
    def __init__(self, feature_dim: int = 1024, token_dim: int = 1024) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, token_dim),
        )
        self.globalPool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, frame_feat: torch.Tensor, key_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        residual = frame_feat - key_feat
        
        pooled = self.globalPool(residual)
        pooled = pooled.flatten(1)

        # this projects to (B, token_dim)
        output = self.mlp(pooled)
        return output
