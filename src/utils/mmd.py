"""
mmd.py
------
Maximum Mean Discrepancy (MMD) for domain gap quantification.

Implements the kernel-based MMD estimator (Gretton et al., 2012) used
in Table 2 of the paper:

    MMD (fused, pre-adaptation)  = 0.142
    MMD (fused, post-adaptation) = 0.067
"""

from __future__ import annotations

import torch


def _rbf_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    bandwidths: list[float] | None = None,
) -> torch.Tensor:
    """
    Multi-bandwidth RBF (Gaussian) kernel matrix between rows of X and Y.

    K(x, y) = Σ_σ exp(−‖x−y‖² / (2σ²))

    Parameters
    ----------
    X : (n, d)
    Y : (m, d)
    bandwidths : list of σ values; defaults to [0.1, 0.5, 1.0, 2.0, 5.0]
    """
    if bandwidths is None:
        bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]

    # Pairwise squared distances: (n, m)
    XX = (X * X).sum(dim=1, keepdim=True)        # (n, 1)
    YY = (Y * Y).sum(dim=1, keepdim=True).T       # (1, m)
    dist2 = XX + YY - 2.0 * X @ Y.T               # (n, m)
    dist2 = dist2.clamp(min=0.0)

    K = torch.zeros_like(dist2)
    for sigma in bandwidths:
        K += torch.exp(-dist2 / (2.0 * sigma ** 2))

    return K


def compute_mmd(
    X: torch.Tensor,
    Y: torch.Tensor,
    bandwidths: list[float] | None = None,
    biased: bool = False,
) -> float:
    """
    Unbiased (or biased) MMD² estimate between samples X (source) and Y (target).

    MMD²(X, Y) = E[k(x,x')] − 2·E[k(x,y)] + E[k(y,y')]

    Parameters
    ----------
    X : (n, d) — source domain embeddings
    Y : (m, d) — target domain embeddings
    biased : if True use biased (faster) estimator

    Returns
    -------
    mmd_value : float (√MMD²)
    """
    n, m = X.shape[0], Y.shape[0]

    Kxx = _rbf_kernel(X, X, bandwidths)
    Kyy = _rbf_kernel(Y, Y, bandwidths)
    Kxy = _rbf_kernel(X, Y, bandwidths)

    if biased:
        mmd2 = Kxx.mean() - 2.0 * Kxy.mean() + Kyy.mean()
    else:
        # Unbiased: exclude diagonal terms from Kxx and Kyy
        mask_xx = 1.0 - torch.eye(n, device=X.device)
        mask_yy = 1.0 - torch.eye(m, device=Y.device)
        mmd2 = (
            (Kxx * mask_xx).sum() / (n * (n - 1))
            - 2.0 * Kxy.mean()
            + (Kyy * mask_yy).sum() / (m * (m - 1))
        )

    return float(mmd2.clamp(min=0.0).sqrt().item())


@torch.no_grad()
def compute_domain_gap(
    model,
    adult_loader,
    neonatal_loader,
    device: str = "cuda",
    max_samples: int = 512,
) -> dict[str, float]:
    """
    Compute per-modality and fused MMD between adult and neonatal embeddings.

    Returns
    -------
    dict with keys: 'video', 'audio', 'physio', 'fused'
    """
    model.eval()

    def _collect(loader, max_n):
        z_v_all, z_a_all, z_p_all, z_f_all = [], [], [], []
        n = 0
        for batch in loader:
            if n >= max_n:
                break
            z_v = model.video_enc(batch["video"].to(device))
            z_a = model.audio_enc(batch["audio"].to(device),
                                  available=batch["audio_available"][0].item())
            z_p = model.physio_enc(batch["physio"].to(device))
            z_f, _ = model.fusion(z_v, z_a, z_p)

            z_v_all.append(z_v.cpu())
            z_a_all.append(z_a.cpu())
            z_p_all.append(z_p.cpu())
            z_f_all.append(z_f.cpu())
            n += z_v.shape[0]

        return (
            torch.cat(z_v_all)[:max_n],
            torch.cat(z_a_all)[:max_n],
            torch.cat(z_p_all)[:max_n],
            torch.cat(z_f_all)[:max_n],
        )

    src_v, src_a, src_p, src_f = _collect(adult_loader,    max_samples)
    tgt_v, tgt_a, tgt_p, tgt_f = _collect(neonatal_loader, max_samples)

    return {
        "video":   compute_mmd(src_v, tgt_v),
        "audio":   compute_mmd(src_a, tgt_a),
        "physio":  compute_mmd(src_p, tgt_p),
        "fused":   compute_mmd(src_f, tgt_f),
    }
