"""Two-tower encoders and the symmetric InfoNCE retrieval loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


def _resnet50_encoder(in_channels: int, pretrained: bool) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    encoder = resnet50(weights=weights)

    if in_channels != 3:
        old_conv = encoder.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            if pretrained:
                mean_weights = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(
                    mean_weights.repeat(1, in_channels, 1, 1)
                )
            else:
                nn.init.kaiming_normal_(
                    new_conv.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

        encoder.conv1 = new_conv

    encoder.fc = nn.Identity()
    return encoder


class ProjectionHead(nn.Module):
    """Non-linear projection head mapping 2048-D features to 512-D metric embeddings.

    Architectural Modes:
        use_batchnorm=False (historical default):
            Linear(2048, 1024) -> ReLU() -> Linear(1024, 512)
            Matches historical best_model-v1.ckpt state-dict keys:
                layers.0.weight, layers.0.bias, layers.2.weight, layers.2.bias
        use_batchnorm=True (optional specification):
            Linear(2048, 1024) -> BatchNorm1d(1024) -> ReLU() -> Linear(1024, 512)
            State-dict keys:
                layers.0.weight, layers.0.bias, layers.1.weight, layers.1.bias,
                layers.1.running_mean, layers.1.running_var, layers.1.num_batches_tracked,
                layers.3.weight, layers.3.bias

    BatchNorm Constraint:
        When use_batchnorm=True, training requires batch size B >= 2 because
        BatchNorm1d cannot compute batch statistics across a single sample.
    """

    def __init__(self, use_batchnorm: bool = False) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm

        if use_batchnorm:
            self.layers = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
            )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class TwoTowerNetwork(nn.Module):
    """Separate SAR and MS ResNet-50 towers with 512-D projections.

    Args:
        pretrained: If True, loads ImageNet pretrained weights for backbones.
        use_batchnorm: If False (default), preserves historical projection head
                       architecture without BatchNorm1d. If True, includes BatchNorm1d.
    """

    def __init__(
        self,
        pretrained: bool = True,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm

        self.sar_encoder = _resnet50_encoder(2, pretrained)
        self.ms_encoder = _resnet50_encoder(3, pretrained)

        self.sar_projection = ProjectionHead(use_batchnorm=use_batchnorm)
        self.ms_projection = ProjectionHead(use_batchnorm=use_batchnorm)

    def forward(
        self,
        sar_batch: Tensor,
        ms_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        sar_embeddings = self.sar_projection(
            self.sar_encoder(sar_batch)
        )
        ms_embeddings = self.ms_projection(
            self.ms_encoder(ms_batch)
        )
        return sar_embeddings, ms_embeddings


class InfoNCELoss(nn.Module):
    """Symmetric in-batch contrastive loss for paired SAR and MS embeddings.

    Computes:
        L_InfoNCE = 0.5 * (CrossEntropy(S, targets) + CrossEntropy(S^T, targets))
    where S = (normalize(Z_sar) @ normalize(Z_ms)^T) / temperature,
    and targets = [0, 1, ..., B-1].

    Properties:
        - Swap Invariance: InfoNCELoss(A, B) == InfoNCELoss(B, A).
        - Theoretical Chance Baseline: When logits are uniform across B candidates,
          loss evaluates to -ln(1/B) = ln(B).
        - Batch Constraint: Requires B >= 2 to provide non-trivial in-batch negative pairs.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        sar_embeddings: Tensor,
        ms_embeddings: Tensor,
    ) -> Tensor:
        if sar_embeddings.ndim != 2 or ms_embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape (batch_size, embedding_dim)"
            )

        if sar_embeddings.shape != ms_embeddings.shape:
            raise ValueError(
                "SAR and MS embeddings must have the same shape"
            )

        batch_size = sar_embeddings.shape[0]
        if batch_size < 2:
            # Contrastive learning requires at least one negative pair per query
            raise ValueError(
                f"InfoNCE requires batch_size >= 2 to form negative pairs, got {batch_size}"
            )

        sar_normalized = F.normalize(sar_embeddings, dim=1)
        ms_normalized = F.normalize(ms_embeddings, dim=1)

        logits = (
            sar_normalized @ ms_normalized.transpose(0, 1)
        ) / self.temperature

        targets = torch.arange(
            batch_size,
            device=logits.device,
        )

        sar_to_ms = self.cross_entropy(logits, targets)
        ms_to_sar = self.cross_entropy(logits.transpose(0, 1), targets)

        return (sar_to_ms + ms_to_sar) / 2.0
