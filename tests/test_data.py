"""
test_data.py
------------
Unit tests for augmentations and episode sampler.
Run with: pytest tests/test_data.py -v
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
import numpy as np


class TestAugmentations:
    def test_video_aug_shape_preserved(self):
        from data.augmentations import VideoAugmentation
        aug = VideoAugmentation()
        x = torch.randn(16, 3, 224, 224)
        out = aug(x)
        assert out.shape == x.shape

    def test_audio_aug_shape_preserved(self):
        from data.augmentations import AudioAugmentation
        aug = AudioAugmentation()
        x = torch.rand(1, 128, 128)
        out = aug(x)
        assert out.shape == x.shape

    def test_physio_aug_shape_preserved(self):
        from data.augmentations import PhysioAugmentation
        aug = PhysioAugmentation()
        x = torch.randn(3, 30)
        out = aug(x)
        assert out.shape == x.shape

    def test_multimodal_two_views_differ(self):
        from data.augmentations import MultimodalAugmentation
        aug = MultimodalAugmentation()
        v = torch.randn(16, 3, 224, 224)
        a = torch.rand(1, 128, 128)
        p = torch.randn(3, 30)
        (v1, a1, p1), (v2, a2, p2) = aug.generate_two_views(v, a, p)
        # Two independent augmentations should differ
        assert not torch.allclose(v1, v2, atol=1e-3)

    def test_audio_skipped_when_unavailable(self):
        from data.augmentations import MultimodalAugmentation
        aug = MultimodalAugmentation()
        v = torch.randn(16, 3, 224, 224)
        a = torch.zeros(1, 128, 128)
        p = torch.randn(3, 30)
        _, a_out, _ = aug(v, a, p, audio_available=False)
        # Zero audio should remain zero when audio_available=False
        assert a_out.abs().sum() == 0.0


class TestEpisodeSampler:
    """
    Uses a minimal in-memory mock dataset to test the sampler.
    """

    class MockDataset:
        def __init__(self, n_subjects=10, n_per_subject=20, n_classes=2):
            self.items = []
            for s in range(n_subjects):
                for i in range(n_per_subject):
                    lbl = i % n_classes
                    self.items.append({
                        "video":           torch.randn(4, 3, 224, 224),
                        "audio":           torch.randn(1, 128, 128),
                        "physio":          torch.randn(3, 30),
                        "label":           lbl,
                        "subject":         f"S{s:03d}",
                        "audio_available": True,
                    })

        def __len__(self): return len(self.items)
        def __getitem__(self, idx): return self.items[idx]

    def test_episode_shapes(self):
        from data.episode_sampler import EpisodeSampler
        ds = self.MockDataset()
        sampler = EpisodeSampler(ds, n_way=2, k_shot=5, n_query=10)
        ep = sampler.sample_episode()

        assert ep.support_labels.shape == (10,)   # 2 * 5
        assert ep.query_labels.shape  == (20,)    # 2 * 10
        assert ep.support_video.shape[0] == 10
        assert ep.query_video.shape[0]   == 20

    def test_labels_in_range(self):
        from data.episode_sampler import EpisodeSampler
        ds = self.MockDataset()
        sampler = EpisodeSampler(ds, n_way=2, k_shot=3, n_query=5)
        ep = sampler.sample_episode()
        assert ep.support_labels.max().item() < 2
        assert ep.query_labels.max().item() < 2
