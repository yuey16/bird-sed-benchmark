"""
Experiment 8: Frame-wise Dynamic Threshold DNN for Acoustic Event Detection

Based on: Xia et al., "Frame-wise dynamic threshold based polyphonic acoustic
event detection", Interspeech 2017.

Approach:
  - Extract mel-filterbank features (128 bins) with 10-frame context
  - Train a 3-layer DNN (500 units each) with BCE loss
  - Compare three thresholding strategies:
    8a. Fixed threshold (sweep on train, best global threshold)
    8b. Contour-based dynamic threshold (alpha * max_prob per frame)
    8c. Regressor-based dynamic threshold (LSTM estimates threshold per frame)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import librosa
from data_utils import load_data, get_audio_and_labels, compute_sed_metrics, FRAME_SHIFT_MS
from sklearn.metrics import f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_MELS = 128
CONTEXT = 5  # frames on each side (total 2*CONTEXT+1 = 11 frames)
HIDDEN = 500
N_EPOCHS_DNN = 30
N_EPOCHS_LSTM = 20
BATCH_SIZE = 4096
LR = 1e-3


# ── Feature Extraction ──

def extract_mel_features(audio, sr):
    """Extract mel-filterbank features aligned to 20ms frames."""
    hop = int(sr * FRAME_SHIFT_MS / 1000)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=2048, hop_length=hop,
        n_mels=N_MELS, fmin=0, fmax=sr // 2
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (N_MELS, T)
    return log_mel.T  # (T, N_MELS)


def add_context(features, context=CONTEXT):
    """Add temporal context: pad and stack neighbouring frames."""
    T, D = features.shape
    padded = np.pad(features, ((context, context), (0, 0)), mode="edge")
    stacked = np.stack([padded[i:i + T] for i in range(2 * context + 1)], axis=1)
    return stacked.reshape(T, -1)  # (T, D * (2*context+1))


def prepare_dataset(dataset):
    """Extract features and labels for all clips in a dataset split."""
    all_feats, all_labels = [], []
    for ex in dataset:
        wav, sr, labels = get_audio_and_labels(ex)
        mel = extract_mel_features(wav, sr)
        mel_ctx = add_context(mel)
        n = min(len(mel_ctx), len(labels))
        all_feats.append(mel_ctx[:n])
        all_labels.append(labels[:n])
    X = np.concatenate(all_feats, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.float32)
    print(f"    Frames: {len(y)}, Feature dim: {X.shape[1]}, Pos rate: {y.mean():.4f}")
    return X, y


def prepare_clip_data(dataset):
    """Return per-clip features and labels (for regressor training)."""
    clips = []
    for ex in dataset:
        wav, sr, labels = get_audio_and_labels(ex)
        mel = extract_mel_features(wav, sr)
        mel_ctx = add_context(mel)
        n = min(len(mel_ctx), len(labels))
        clips.append((mel_ctx[:n].astype(np.float32), labels[:n].astype(np.float32)))
    return clips


# ── DNN Classifier (Section 2 of paper) ──

class DNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dnn(X_train, y_train, input_dim):
    model = DNNClassifier(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model.train()
    for epoch in range(N_EPOCHS_DNN):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{N_EPOCHS_DNN}, Loss: {total_loss/len(X_train):.4f}")

    return model


def predict_probs(model, X):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 4, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


# ── 8a: Fixed Threshold ──

def find_best_fixed_threshold(probs, labels):
    best_f1, best_t = -1, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


# ── 8b: Contour-based Dynamic Threshold (Section 3.1) ──

def contour_threshold(probs_per_clip, labels_per_clip, alpha):
    """Apply contour-based dynamic threshold: T_k = alpha * prob_k."""
    all_preds, all_labels = [], []
    for probs, labels in zip(probs_per_clip, labels_per_clip):
        thresholds = alpha * probs
        preds = (probs >= thresholds).astype(int)
        all_preds.append(preds)
        all_labels.append(labels)
    return np.concatenate(all_labels), np.concatenate(all_preds)


# ── 8c: Regressor-based Dynamic Threshold (Section 3.2) ──

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden=50):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out)).squeeze(-1)


def train_lstm_regressor(clips_feats, clips_probs, dnn_model, input_dim):
    """Train LSTM to predict normalized probability threshold per frame."""
    seq_inputs, seq_targets = [], []

    for feat, labels in zip(clips_feats, clips_probs):
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = dnn_model(feat_t.squeeze(0))
            probs = torch.sigmoid(logits).cpu().numpy()

        correct_mask = ((probs >= 0.5).astype(int) == labels.astype(int))
        if correct_mask.sum() < 2:
            continue

        p_correct = probs[correct_mask]
        fmax, fmin = p_correct.max(), p_correct.min()
        if fmax - fmin < 1e-6:
            continue

        U = (probs - fmin) / (fmax - fmin)
        U = np.clip(U, 0, 1)
        seq_inputs.append(feat)
        seq_targets.append(U.astype(np.float32))

    regressor = LSTMRegressor(input_dim).to(DEVICE)
    optimizer = torch.optim.SGD(regressor.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    regressor.train()
    for epoch in range(N_EPOCHS_LSTM):
        total_loss = 0
        for feat, target in zip(seq_inputs, seq_targets):
            x = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)
            y = torch.from_numpy(target).unsqueeze(0).to(DEVICE)
            pred = regressor(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"      LSTM Epoch {epoch+1}/{N_EPOCHS_LSTM}, Loss: {total_loss/len(seq_inputs):.6f}")

    return regressor


def regressor_threshold(clips, dnn_model, regressor, input_dim):
    """Apply regressor-based dynamic threshold on clips."""
    all_preds, all_labels = [], []
    dnn_model.eval()
    regressor.eval()

    for feat, labels in clips:
        feat_t = torch.from_numpy(feat).to(DEVICE)
        with torch.no_grad():
            logits = dnn_model(feat_t)
            probs = torch.sigmoid(logits).cpu().numpy()

            x_seq = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)
            U_hat = regressor(x_seq).squeeze(0).cpu().numpy()

        fmax, fmin = probs.max(), probs.min()
        dynamic_thresh = fmin + U_hat * (fmax - fmin)
        preds = (probs >= dynamic_thresh).astype(int)
        all_preds.append(preds)
        all_labels.append(labels)

    return np.concatenate(all_labels), np.concatenate(all_preds)


def main():
    print("=" * 60)
    print("EXP 8: Frame-wise Dynamic Threshold DNN (Xia et al. 2017)")
    print("=" * 60)

    train_ds, test_ds = load_data()

    print("\n--- Extracting features ---")
    print("  Train:")
    X_train, y_train = prepare_dataset(train_ds)
    print("  Test:")
    X_test, y_test = prepare_dataset(test_ds)

    input_dim = X_train.shape[1]

    print(f"\n--- Training DNN classifier on {DEVICE} ---")
    dnn_model = train_dnn(X_train, y_train, input_dim)

    train_probs = predict_probs(dnn_model, X_train)
    test_probs = predict_probs(dnn_model, X_test)

    # 8a: Fixed threshold
    print("\n  === 8a: Fixed Threshold ===")
    best_t, best_f1_train = find_best_fixed_threshold(train_probs, y_train)
    print(f"    Best threshold: {best_t:.2f} (train F1={best_f1_train:.4f})")
    print("\n    --- Train ---")
    compute_sed_metrics(y_train, (train_probs >= best_t).astype(int))
    print("\n    --- Test ---")
    m_fixed = compute_sed_metrics(y_test, (test_probs >= best_t).astype(int))

    # 8b: Contour-based
    print("\n  === 8b: Contour-based Dynamic Threshold ===")
    alpha = y_train.mean()
    print(f"    Alpha (active frame ratio): {alpha:.4f}")

    train_clips = prepare_clip_data(train_ds)
    test_clips = prepare_clip_data(test_ds)

    train_clips_probs = []
    for feat, labels in train_clips:
        p = predict_probs(dnn_model, feat)
        train_clips_probs.append(p)
    test_clips_probs = []
    for feat, labels in test_clips:
        p = predict_probs(dnn_model, feat)
        test_clips_probs.append(p)

    labels_cat, preds_cat = contour_threshold(
        train_clips_probs, [l for _, l in train_clips], alpha
    )
    print("\n    --- Train ---")
    compute_sed_metrics(labels_cat, preds_cat)

    labels_cat, preds_cat = contour_threshold(
        test_clips_probs, [l for _, l in test_clips], alpha
    )
    print("\n    --- Test ---")
    m_contour = compute_sed_metrics(labels_cat, preds_cat)

    # 8c: Regressor-based
    print("\n  === 8c: Regressor-based Dynamic Threshold (LSTM) ===")
    print("    Training LSTM regressor...")
    regressor = train_lstm_regressor(
        [f for f, _ in train_clips],
        [l for _, l in train_clips],
        dnn_model, input_dim
    )

    print("\n    --- Train ---")
    labels_r, preds_r = regressor_threshold(train_clips, dnn_model, regressor, input_dim)
    compute_sed_metrics(labels_r, preds_r)

    print("\n    --- Test ---")
    labels_r, preds_r = regressor_threshold(test_clips, dnn_model, regressor, input_dim)
    m_reg = compute_sed_metrics(labels_r, preds_r)

    print(f"\n{'=' * 60}")
    print(f"FINAL -- DNN Fixed Threshold   Test F1: {m_fixed['f1']:.4f}")
    print(f"FINAL -- DNN Contour Threshold Test F1: {m_contour['f1']:.4f}")
    print(f"FINAL -- DNN LSTM Regressor    Test F1: {m_reg['f1']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
