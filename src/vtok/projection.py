import torch
import torch.nn as nn

class VisualProjection(nn.Module):
    """
    Projection of visual tokens generated from VTok to the MLLM's embedding space.
    """
    def __init__(self, token_dimension: int = 1024, model_dim: int = 4096) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dimension, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the visual projector.
        Args:
            tokens: The tokens (B, L_v, d_v) generated from the Vtok tokeniser.
        Returns:
            embedding (B, L_v, d_m) into the language space (SES).
        """
        x = self.mlp(tokens)
        return x
