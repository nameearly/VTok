from dataclasses import dataclass

@dataclass
class VTokConfig:
    # feature extractor
    backbone: str = "vgg19" # choose from vgg19 or clip
    vgg_layer_index: int = 25
    freeze_backbone: bool = True

    # device type
    device: str = "cuda"

    # spaitla encoder
    spatial_grid_size: int = 4
    token_dim: int = 512
    # motion encoder
    temporal_stride: int = 6
    # key frame index
    key_frame_index: int = 0 # typify which frame is key.