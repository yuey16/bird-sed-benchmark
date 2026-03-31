"""
Experiments 9a-9d: Simple temporal-regularisation control for BEATs-based SED

  9a: BEATs + Linear  + Total Variation (TV) smoothing
  9b: BEATs + Linear  + Hysteresis thresholding
  9c: BEATs + CRNN    + TV smoothing
  9d: BEATs + CRNN    + Hysteresis thresholding

BEATs frame-level features are extracted once and shared across all variants.
Two classifiers (Linear / CRNN) produce per-frame probabilities, then two
temporal post-processing methods (TV / Hysteresis) are applied on top.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.interpolate import interp1d
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BEATs import BEATs, BEATsConfig
from data_utils import load_data, get_audio_and_labels, compute_sed_metrics, FRAME_SHIFT_MS

BEATS_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "BEATs_iter3_plus_AS2M.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -- BEATs feature extraction (shared) --

def load_beats_model():
    ckpt = torch.load(BEATS_CKPT, map_location="cpu")
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(DEVICE)
    return model


def extract_clip_features(beats_model, dataset):
    """Return list of (features, labels) per clip. features: (T, 768)."""
    clips = []
    for i, ex in enumerate(dataset):
        wav, sr, labels = get_audio_and_labels(ex)
        n_frames = len(labels)

        t = torch.from_numpy(wav).float().unsqueeze(0)
        if sr != 16000:
            t = torchaudio.functional.resample(t, orig_freq=sr, new_freq=16000)
        t = t.to(DEVICE)

        with torch.no_grad():
            feats, _ = beats_model.extract_features(t)
        feats = feats.squeeze(0).cpu().numpy()

        if feats.shape[0] != n_frames:
            x_o = np.linspace(0, 1, feats.shape[0])
            x_t = np.linspace(0, 1, n_frames)
            feats = interp1d(x_o, feats, axis=0,
                             kind="linear", fill_value="extrapolate")(x_t)

        clips.append((feats.astype(np.float32), labels))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(dataset)} clips...")

    print(f"    Done: {len(clips)} clips, "
          f"total frames={sum(l.shape[0] for _, l in clips)}")
    return clips


# -- Temporal-regularisation primitives --

def tv_denoise_1d(y, lam):
    """Solve  min_x  0.5*||x-y||^2 + lam*TV(x),  x in [0,1].
    Uses Huber-smoothed TV with L-BFGS-B."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 2 or lam <= 0:
        return y.copy()
    eps = 1e-4

    def obj_grad(x):
        diff = x - y
        obj = 0.5 * np.dot(diff, diff)
        grad = diff.copy()
        dx = np.diff(x)
        a = np.sqrt(dx * dx + eps)
        obj += lam * a.sum()
        g = dx / a
        grad[:-1] += lam * g
        grad[1:] -= lam * g
        return obj, grad

    res = minimize(obj_grad, y, jac=True, method="L-BFGS-B",
                   bounds=[(0.0, 1.0)] * n,
                   options={"maxiter": 60, "ftol": 1e-8})
    return res.x


def hysteresis_threshold(probs, t_high, t_low):
    """Two-threshold hysteresis: activate when prob >= t_high,
    deactivate when prob < t_low."""
    preds = np.zeros(len(probs), dtype=np.int32)
    active = False
    for i in range(len(probs)):
        if active:
            if probs[i] < t_low:
                active = False
            else:
                preds[i] = 1
        else:
            if probs[i] >= t_high:
                active = True
                preds[i] = 1
    return preds


# -- Apply to clip lists --

def apply_tv_clips(clip_probs, clip_labels, lam):
    all_l, all_p = [], []
    for probs, labels in zip(clip_probs, clip_labels):
        s = tv_denoise_1d(probs, lam)
        all_p.append((s >= 0.5).astype(np.int32))
        all_l.append(labels)
    return np.concatenate(all_l), np.concatenate(all_p)


def apply_hyst_clips(clip_probs, clip_labels, t_high, t_low):
    all_l, all_p = [], []
    for probs, labels in zip(clip_probs, clip_labels):
        all_p.append(hysteresis_threshold(probs, t_high, t_low))
        all_l.append(labels)
    return np.concatenate(all_l), np.concatenate(all_p)


# -- Hyper-parameter tuning --

def tune_tv(clip_probs, clip_labels):
    best_lam, best_f1 = 0.01, 0
    for lam in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        l, p = apply_tv_clips(clip_probs, clip_labels, lam)
        f = f1_score(l, p, zero_division=0)
        print(f"      lam={lam:.3f}  F1={f:.4f}")
        if f > best_f1:
            best_f1, best_lam = f, lam
    print(f"    -> best lam={best_lam:.3f}  (train F1={best_f1:.4f})")
    return best_lam


def tune_hyst(clip_probs, clip_labels):
    best, best_f1 = (0.5, 0.3), 0
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for tl in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            if tl >= th:
                continue
            l, p = apply_hyst_clips(clip_probs, clip_labels, th, tl)
            f = f1_score(l, p, zero_division=0)
            if f > best_f1:
                best_f1, best = f, (th, tl)
    print(f"    -> best t_high={best[0]:.2f}, t_low={best[1]:.2f}  "
          f"(train F1={best_f1:.4f})")
    return best


# -- Linear classifier --

def train_linear(clips):
    X = np.concatenate([f for f, _ in clips])
    y = np.concatenate([l for _, l in clips])
    sc = StandardScaler()
    X = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X, y)
    return clf, sc


def linear_clip_probs(clf, sc, clips):
    cp, cl = [], []
    for feats, labels in clips:
        p = clf.predict_proba(sc.transform(feats))[:, 1]
        cp.append(p)
        cl.append(labels)
    return cp, cl


# -- CRNN classifier --

class CRNN(nn.Module):
    def __init__(self, in_dim=768, ch=128, gru_h=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2),     nn.BatchNorm1d(ch), nn.ReLU(),
        )
        self.gru = nn.GRU(ch, gru_h, batch_first=True, bidirectional=True)
        self.head = nn.Linear(gru_h * 2, 1)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, _ = self.gru(x)
        return self.head(x).squeeze(-1)


def train_crnn(clips, n_epochs=30, lr=1e-3, bs=16):
    dim = clips[0][0].shape[1]
    maxlen = max(f.shape[0] for f, _ in clips)

    model = CRNN(in_dim=dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    model.train()
    for ep in range(n_epochs):
        idx = np.random.RandomState(ep).permutation(len(clips))
        ep_loss, nb = 0.0, 0

        for s in range(0, len(idx), bs):
            bi = idx[s:s + bs]
            B = len(bi)
            Xb = np.zeros((B, maxlen, dim), dtype=np.float32)
            yb = np.zeros((B, maxlen), dtype=np.float32)
            mk = np.zeros((B, maxlen), dtype=np.float32)
            for j, k in enumerate(bi):
                f, l = clips[k]
                T = f.shape[0]
                Xb[j, :T] = f
                yb[j, :T] = l
                mk[j, :T] = 1.0

            Xt = torch.from_numpy(Xb).to(DEVICE)
            yt = torch.from_numpy(yb).to(DEVICE)
            mt = torch.from_numpy(mk).to(DEVICE)

            logits = model(Xt)
            loss = (bce(logits, yt) * mt).sum() / mt.sum()

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1

        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    Epoch {ep+1}/{n_epochs}  loss={ep_loss / nb:.4f}")

    model.eval()
    return model


def crnn_clip_probs(model, clips):
    cp, cl = [], []
    model.eval()
    with torch.no_grad():
        for feats, labels in clips:
            X = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)
            logits = model(X).squeeze(0).cpu().numpy()
            p = 1.0 / (1.0 + np.exp(-logits))
            cp.append(p[:len(labels)])
            cl.append(labels)
    return cp, cl


# -- Run one variant --

def run_variant(tag, probs_tr, labs_tr, probs_te, labs_te, method):
    print(f"\n  === {tag} ===")
    if method == "tv":
        print("    Tuning TV lambda on train...")
        lam = tune_tv(probs_tr, labs_tr)
        print("\n    Train:")
        l, p = apply_tv_clips(probs_tr, labs_tr, lam)
        compute_sed_metrics(l, p)
        print("\n    Test:")
        l, p = apply_tv_clips(probs_te, labs_te, lam)
        return compute_sed_metrics(l, p)
    else:
        print("    Tuning hysteresis on train...")
        th, tl = tune_hyst(probs_tr, labs_tr)
        print("\n    Train:")
        l, p = apply_hyst_clips(probs_tr, labs_tr, th, tl)
        compute_sed_metrics(l, p)
        print("\n    Test:")
        l, p = apply_hyst_clips(probs_te, labs_te, th, tl)
        return compute_sed_metrics(l, p)


# -- Main --

def main():
    print("=" * 60)
    print("EXP 9: BEATs + Temporal Regularisation (4 variants)")
    print("=" * 60)

    train_ds, test_ds = load_data()

    print("\nLoading BEATs...")
    beats = load_beats_model()
    print(f"BEATs on {DEVICE}")

    print("\n--- Extracting train features ---")
    tr_clips = extract_clip_features(beats, train_ds)
    print("\n--- Extracting test features ---")
    te_clips = extract_clip_features(beats, test_ds)

    del beats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 9a / 9b: Linear classifier
    print("\n" + "-" * 60)
    print("Training Linear classifier (Logistic Regression)...")
    clf, sc = train_linear(tr_clips)
    print("  Done.")

    lp_tr, ll_tr = linear_clip_probs(clf, sc, tr_clips)
    lp_te, ll_te = linear_clip_probs(clf, sc, te_clips)

    print("\n  --- Linear baseline (threshold=0.5, no temporal reg) ---")
    base_l = np.concatenate(ll_te)
    base_p = np.concatenate([(p >= 0.5).astype(np.int32) for p in lp_te])
    m_lin_base = compute_sed_metrics(base_l, base_p)

    m_9a = run_variant("9a: BEATs + Linear + TV",
                       lp_tr, ll_tr, lp_te, ll_te, "tv")
    m_9b = run_variant("9b: BEATs + Linear + Hysteresis",
                       lp_tr, ll_tr, lp_te, ll_te, "hysteresis")

    # 9c / 9d: CRNN classifier
    print("\n" + "-" * 60)
    print("Training CRNN classifier (Conv1D + BiGRU)...")
    crnn = train_crnn(tr_clips)
    print("  Done.")

    cp_tr, cl_tr = crnn_clip_probs(crnn, tr_clips)
    cp_te, cl_te = crnn_clip_probs(crnn, te_clips)

    print("\n  --- CRNN baseline (threshold=0.5, no temporal reg) ---")
    base_l2 = np.concatenate(cl_te)
    base_p2 = np.concatenate([(p >= 0.5).astype(np.int32) for p in cp_te])
    m_crnn_base = compute_sed_metrics(base_l2, base_p2)

    m_9c = run_variant("9c: BEATs + CRNN + TV",
                       cp_tr, cl_tr, cp_te, cl_te, "tv")
    m_9d = run_variant("9d: BEATs + CRNN + Hysteresis",
                       cp_tr, cl_tr, cp_te, cl_te, "hysteresis")

    # Summary
    sep = "=" * 60
    print(f"\n{sep}")
    print("SUMMARY -- Temporal Regularisation")
    print(sep)
    print(f"  Linear baseline (no reg)       Test F1: {m_lin_base['f1']:.4f}")
    print(f"  9a  BEATs + Linear + TV        Test F1: {m_9a['f1']:.4f}")
    print(f"  9b  BEATs + Linear + Hyst      Test F1: {m_9b['f1']:.4f}")
    print(f"  CRNN baseline (no reg)         Test F1: {m_crnn_base['f1']:.4f}")
    print(f"  9c  BEATs + CRNN  + TV         Test F1: {m_9c['f1']:.4f}")
    print(f"  9d  BEATs + CRNN  + Hyst       Test F1: {m_9d['f1']:.4f}")
    print(sep)


if __name__ == "__main__":
    main()
