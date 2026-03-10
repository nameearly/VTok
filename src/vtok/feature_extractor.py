import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPVisionModel

class VGGFeatureExtractor(nn.Module):
    """
    Returns the features at a configurable level from the VGG19 model.
    """
    def __init__(self, layer_index: int = 25, freeze: bool = True) -> None:
        """
        Initialise the VGGFeatureExtractor
        """
        super().__init__()
        self.freeze = freeze
        VGG = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # self.feature_extractor = nn.Sequential(*list(VGG.children())[:layer_index])
        self.feature_extractor = VGG.features[:layer_index]
        if freeze:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            self.feature_extractor.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 3, H, W) normalised Image(s).
        Returns:
            (Batch, C_feat, H', W') feature map.
        """
        features = self.feature_extractor(x)
        return features
    def train(self, mode: bool = True):
        super().train(mode=mode)
        if self.freeze:
            self.feature_extractor.eval()
        return self

class ClipFeatureExtractor(nn.Module):
    """
    Uses the vision transformer from openai/clip-vit-large-patch14-336
    """
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14-336", freeze: bool = True):
        """
        Initialise the Clip Feature Extractor
        """
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_id)
        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
    
    def train(self, mode: bool = True):
        super().train(mode=mode)
        if self.freeze:
            self.model.eval()
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 3, 336, 336) preprocessed images.
        Returns:
            (Batch, 1024, 24, 24)
        """
        # process the image.
        outputs = self.model(x)
        features = outputs.last_hidden_state
        features = features[:, 1:] # remove CLS token.

        batch, patch, dim = features.shape
        grid_size = int(patch ** 0.5)
        features = features.reshape(batch, grid_size, grid_size, dim)
        features = features.permute(0, 3, 1, 2).contiguous()
        return features