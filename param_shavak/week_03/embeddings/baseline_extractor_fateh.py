import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# ---------------------------------------------------------------------------
# NON-CANONICAL ALTERNATIVE — preserved for historical reference.
#
# This class (BaselineExtractor) was NOT used to generate the historical
# baseline embeddings in results/local_pc/week_03/.
#
# The canonical SAR baseline encoder is ResNet50S1Encoder from
# weeks/week_04/models/resnet_meet.py (Strategy B: copy ch0+ch1,
# distribute ch2 evenly). The canonical MS baseline encoder is the
# standard 3-channel ResNet50 used in
# weeks/week_03/embeddings/generate_baseline_embeddings_adarsh.py.
#
# SAR channel adapter used here: Strategy A (copy first in_channels
# columns from pretrained RGB filter — Blue channel discarded).
# This differs from the Two-Tower's Strategy C (mean-RGB repeated)
# and from the canonical Strategy B.
#
# See: shared/docs/audit_report/week3_scientific_clarification_report.md
#      Section 1.3 for a full mathematical comparison of all strategies.
# ---------------------------------------------------------------------------


class BaselineExtractor(nn.Module):
    """
    Pretrained ResNet50 with FC removed.
    Outputs 2048-dimensional embeddings.
    Used as zero-shot baseline before contrastive learning.

    NON-CANONICAL ALTERNATIVE. See module docstring above.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        base = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        if in_channels != 3:
            old_conv = base.conv1

            new_conv = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

            with torch.no_grad():
                for i in range(in_channels):
                    new_conv.weight[:, i, :, :] = old_conv.weight[:, i, :, :]

            base.conv1 = new_conv

        # Remove ResNet classification layer
        base.fc = nn.Identity()

        self.encoder = base

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":

    # Sentinel-2 RGB test
    s2_model = BaselineExtractor(in_channels=3)

    out = s2_model(
        torch.randn(4, 3, 224, 224)
    )

    assert out.shape == (4, 2048)

    print(f"S2 extractor: {out.shape} OK")

    # SAR test
    sar_model = BaselineExtractor(in_channels=2)

    out = sar_model(
        torch.randn(4, 2, 224, 224)
    )

    assert out.shape == (4, 2048)

    print(f"SAR extractor: {out.shape} OK")