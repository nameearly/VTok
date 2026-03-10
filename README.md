# VTok — Unofficial Implementation

Unofficial PyTorch implementation of [VTok: A Unified Video Tokenizer with Decoupled Spatial-Temporal Latents](https://arxiv.org/pdf/2602.04202) (Wang et al., 2026).

VTok reduces video token complexity from O(T × S) to O(T + S) by decoupling spatial and temporal representations: retain the full spatial features of a single key frame, encode each subsequent frame as a compact residual motion token.

## What's implemented

**Tokeniser (complete):**

- `VTokTokeniser` — full video tokenisation pipeline producing `(B, S + T_motion, d_v)` token sequences
- Pluggable feature extraction backbone — VGG19 (`conv4_4`, 512-dim) or CLIP-L/336px (1024-dim)
- `SpatialEncoder` — adaptive pooling of key frame features into a configurable S-token grid (default 4×4 = 16 tokens)
- `MotionEncoder` — computes per-frame residual `F(x_t) - F(x_key)`, globally pools, and projects via a learned 2-layer MLP (`g_φ` in the paper)
- Configurable temporal stride (default: 6 frames per motion token) and key frame selection

**Unified framework (in progress):**

- `VTokFramework` — understanding and generation branches within a shared autoregressive MLLM (LLaVA-Next / LLaMA-3-8B)
- Visual projection head (`φ_vis`) mapping tokeniser output to MLLM embedding space
- HunyuanVideo diffusion transformer + VAE decoder (frozen)
- Training loop with the combined objective from equation (12): `L = L_under + λ_visLM · L_visLM + λ_dec · L_dec`

## Structure

```
vtok/
├── __init__.py
├── config.py                  # Typed dataclass configuration
├── feature_extractor.py       # VGG19 + CLIP-L backbones
├── spatial_encoder.py         # Key frame → S spatial tokens
├── motion_encoder.py          # Frame residual → single motion token
├── tokeniser.py               # Orchestrates the full pipeline
├── projection.py              # Visual → MLLM embedding projection
├── framework.py               # Unified understanding + generation model
├── train.py                   # Training loop
└── data/
    └── dataset.py             # Video-caption dataset
```

## Installation

```bash
pip install torch torchvision transformers diffusers
```

For CLIP backbone support:
```bash
pip install transformers[torch]
```

## Quick start

```python
from vtok.config import VTokConfig
from vtok.tokeniser import VTokTokeniser
import torch

config = VTokConfig(backbone="clip", spatial_grid_size=4, token_dim=1024)
tokeniser = VTokTokeniser(config)

# (batch, frames, channels, height, width)
video = torch.randn(1, 30, 3, 336, 336)
tokens = tokeniser(video)
# tokens.shape: (1, 16 + 4, 1024) = (1, 20, 1024)
# 16 spatial tokens (4x4 grid) + 4 motion tokens (30 frames / stride 6, minus key frame)
```

With VGG19 backbone:
```python
config = VTokConfig(backbone="vgg19", vgg_layer_index=25, spatial_grid_size=4, token_dim=512)
tokeniser = VTokTokeniser(config)

video = torch.randn(1, 30, 3, 224, 224)
tokens = tokeniser(video)
# tokens.shape: (1, 20, 512)
```

## Design decisions

**Backbone choice.** The paper uses CLIP-L/336px as the shared feature extractor `F` and key frame encoder `E_key` (they are the same model with the same weights — see Section 3.2). We additionally support VGG19 for lighter-weight experimentation. The tokenisation paradigm is backbone-agnostic; the SpatialEncoder and MotionEncoder consume `(B, C_feat, H', W')` feature maps regardless of source.

**Key frame selection.** The paper uses the first frame by default and notes that a dedicated key frame detector yields slight improvements. We expose `key_frame_idx` as a parameter on both the config and the forward call, so custom selection logic can be implemented externally.

**Motion encoder architecture.** The paper specifies `g_φ` as a learned projection from pooled residual features to the motion token space. We implement this as a 2-layer MLP with GELU activation, which is standard for projection heads in vision-language models.

## Differences from the paper

- We support VGG19 as an alternative backbone (paper uses CLIP-L exclusively).
- The unified framework and training loop are work-in-progress.
- We do not yet implement the TV-Align evaluation benchmark.

## Citation

```bibtex
@article{wang2026vtok,
  title={VTok: A Unified Video Tokenizer with Decoupled Spatial-Temporal Latents},
  author={Wang, Feng and Shi, Yichun and Yang, Ceyuan and Guo, Qiushan and Sun, Jingxiang and Yuille, Alan and Wang, Peng},
  journal={arXiv preprint arXiv:2602.04202},
  year={2026}
}
```

## Licence

This is an unofficial implementation for research purposes. The original work is by Bytedance Seed and Johns Hopkins University.