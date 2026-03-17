"""
test_models.py
--------------
Unit tests for encoder, fusion, and full model forward passes.
Run with: pytest tests/test_models.py -v
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B = 2   # batch size
T = 4   # temporal frames (small for test speed)

@pytest.fixture
def video():
    return torch.randn(B, T, 3, 224, 224)

@pytest.fixture
def audio():
    return torch.randn(B, 1, 128, 128)

@pytest.fixture
def physio():
    return torch.randn(B, 3, 30)


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------

class TestVideoEncoder:
    def test_output_shape(self, video):
        from models.encoders import VideoEncoder
        enc = VideoEncoder(pretrained=False)
        out = enc(video)
        assert out.shape == (B, 768), f"Expected (B, 768), got {out.shape}"

    def test_unavailable_returns_token(self, video):
        from models.encoders import VideoEncoder
        enc = VideoEncoder(pretrained=False)
        out = enc(video, available=False)
        assert out.shape == (B, 768)

    def test_projection_l2_normalized(self, video):
        from models.encoders import VideoEncoder
        enc = VideoEncoder(pretrained=False)
        proj = enc.get_projection(video)
        norms = proj.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)


class TestAudioEncoder:
    def test_output_shape(self, audio):
        from models.encoders import AudioEncoder
        enc = AudioEncoder()
        out = enc(audio)
        assert out.shape == (B, 512)

    def test_unavailable(self, audio):
        from models.encoders import AudioEncoder
        enc = AudioEncoder()
        out = enc(audio, available=False)
        assert out.shape == (B, 512)


class TestPhysioEncoder:
    def test_output_shape(self, physio):
        from models.encoders import PhysioEncoder
        enc = PhysioEncoder()
        out = enc(physio)
        assert out.shape == (B, 256)


# ---------------------------------------------------------------------------
# Fusion tests
# ---------------------------------------------------------------------------

class TestMultimodalFusion:
    def test_output_shape(self, video, audio, physio):
        from models.encoders import VideoEncoder, AudioEncoder, PhysioEncoder
        from models.fusion import MultimodalFusion
        z_v = VideoEncoder(pretrained=False)(video)
        z_a = AudioEncoder()(audio)
        z_p = PhysioEncoder()(physio)

        fusion = MultimodalFusion(n_layers=2)   # small for speed
        z_f, attn = fusion(z_v, z_a, z_p)

        assert z_f.shape == (B, 768)
        assert len(attn) == 2

    def test_attention_shape(self, video, audio, physio):
        from models.encoders import VideoEncoder, AudioEncoder, PhysioEncoder
        from models.fusion import MultimodalFusion
        z_v = VideoEncoder(pretrained=False)(video)
        z_a = AudioEncoder()(audio)
        z_p = PhysioEncoder()(physio)

        fusion = MultimodalFusion(n_layers=2)
        _, attn = fusion(z_v, z_a, z_p)

        # Each attention tensor should be (B, 3, 3)
        for a in attn:
            assert a.shape == (B, 3, 3), f"Expected (B,3,3), got {a.shape}"


# ---------------------------------------------------------------------------
# Prototypical network tests
# ---------------------------------------------------------------------------

class TestPrototypicalHead:
    def test_log_probs_shape(self):
        from models.prototypical import PrototypicalHead
        N, K, Q, D = 2, 5, 10, 768
        z_sup = torch.randn(N * K, D)
        y_sup = torch.tensor([0] * K + [1] * K)
        z_qry = torch.randn(Q, D)

        head = PrototypicalHead()
        log_p = head(z_sup, y_sup, z_qry)
        assert log_p.shape == (Q, N)

    def test_loss_backward(self):
        from models.prototypical import PrototypicalHead, prototypical_loss
        N, K, Q, D = 2, 3, 6, 128
        z_sup = torch.randn(N * K, D, requires_grad=True)
        y_sup = torch.tensor([0] * K + [1] * K)
        z_qry = torch.randn(Q, D)
        y_qry = torch.tensor([0, 0, 0, 1, 1, 1])

        head = PrototypicalHead()
        log_p = head(z_sup, y_sup, z_qry)
        loss, acc = prototypical_loss(log_p, y_qry)
        loss.backward()
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Full model tests
# ---------------------------------------------------------------------------

class TestFullModel:
    @pytest.fixture
    def model(self):
        from models.full_model import MultimodalPainModel
        return MultimodalPainModel(video_pretrained=False, n_fusion_layers=2)

    def test_forward(self, model, video, audio, physio):
        N, K, Q = 2, 2, 4
        sup_v = video[:N*K]; sup_a = audio[:N*K]; sup_p = physio[:N*K]
        qry_v = video[:Q];   qry_a = audio[:Q];   qry_p = physio[:Q]
        sup_y = torch.tensor([0, 0, 1, 1])
        qry_y = torch.tensor([0, 0, 1, 1])

        out = model(sup_v, sup_a, sup_p, sup_y, qry_v, qry_a, qry_p, qry_y)

        assert "loss" in out
        assert "predictions" in out
        assert out["log_probs"].shape == (Q, N)

    def test_missing_audio(self, model, video, audio, physio):
        N, K, Q = 2, 2, 4
        sup_v = video[:N*K]; sup_a = torch.zeros_like(audio[:N*K]); sup_p = physio[:N*K]
        qry_v = video[:Q];   qry_a = torch.zeros_like(audio[:Q]);   qry_p = physio[:Q]
        sup_y = torch.tensor([0, 0, 1, 1])
        qry_y = torch.tensor([0, 0, 1, 1])

        out = model(
            sup_v, sup_a, sup_p, sup_y,
            qry_v, qry_a, qry_p, qry_y,
            modality_mask={"video": True, "audio": False, "physio": True},
        )
        assert "loss" in out

    def test_get_embeddings(self, model, video, audio, physio):
        z = model.get_embeddings(video, audio, physio)
        assert z.shape == (B, 768)
