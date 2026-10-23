"""Small CNN classifier for images. Used two ways, always trained on REAL training data only:
   - as the evaluation feature extractor (its penultimate layer defines the image evaluation space,
     a domain-specific stand-in for the Inception features behind FID), and
   - as the downstream model in the class-balancing / augmentation test (real vs real+synthetic).
"""
import torch
import torch.nn as nn

from models.conv import check_shape, to_nchw


class SmallCNN(nn.Module):
    def __init__(self, image_shape, n_classes, base=32, feat_dim=128):
        super().__init__()
        check_shape(image_shape)
        h, w, c = image_shape
        self.shape = tuple(image_shape)
        self.body = nn.Sequential(
            nn.Conv2d(c, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(base, 2 * base, 3, padding=1), nn.BatchNorm2d(2 * base), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * base, 4 * base, 3, padding=1), nn.BatchNorm2d(4 * base), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(4 * base, feat_dim), nn.ReLU())
        self.head = nn.Linear(feat_dim, n_classes)

    def features(self, x):
        return self.body(to_nchw(x, self.shape))  # (batch, feat_dim)

    def forward(self, x):
        return self.head(self.features(x))
