import argparse
import urllib.request
from datetime import datetime
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import AudioCNN

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ESC50_DIR = DATA_DIR / "ESC-50-master"
MODEL_DIR = ROOT / "models"
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

SAMPLE_RATE = 22050


def download_esc50():
    if (ESC50_DIR / "meta" / "esc50.csv").exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "esc50.zip"
    if not zip_path.exists():
        print("Downloading ESC-50 (~600 MB)...")
        with tqdm(unit="B", unit_scale=True) as bar:
            def hook(blocks, block_size, total):
                bar.total = total
                bar.update(blocks * block_size - bar.n)
            urllib.request.urlretrieve(ESC50_URL, zip_path, reporthook=hook)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()


class ESC50Dataset(Dataset):
    """Returns raw mono waveforms; spectrograms are computed on the GPU in batches."""

    def __init__(self, data_dir, metadata_file, split="train"):
        df = pd.read_csv(metadata_file)
        # Fold 5 is held out for testing, as in the video
        self.df = df[df["fold"] != 5] if split == "train" else df[df["fold"] == 5]
        self.df = self.df.reset_index(drop=True)
        self.audio_dir = Path(data_dir) / "audio"
        self.classes = sorted(df["category"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = sf.read(self.audio_dir / row["filename"], dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return torch.from_numpy(waveform), sr, self.class_to_idx[row["category"]]


def collate(batch):
    waveforms, srs, labels = zip(*batch)
    return torch.stack(waveforms), srs[0], torch.tensor(labels)


def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - install the CUDA build of PyTorch first.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Using {torch.cuda.get_device_name(0)}")

    download_esc50()
    MODEL_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(MODEL_DIR / "tensorboard_logs" / f"run_{timestamp}")

    to_mel = nn.Sequential(
        T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512,
                         n_mels=128, f_min=0, f_max=11025),
        T.AmplitudeToDB(),
    ).to(device)
    spec_augment = nn.Sequential(
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80),
    ).to(device)

    def to_spectrogram(waveforms, sr, train):
        # Like the original repo, the 44.1 kHz audio is not resampled before the
        # 22050 Hz mel transform, so spectrograms match the repo's inference code.
        waveforms = waveforms.to(device, non_blocking=True)
        spec = to_mel(waveforms).unsqueeze(1)  # (B, 1, n_mels, time)
        return spec_augment(spec) if train else spec

    meta = ESC50_DIR / "meta" / "esc50.csv"
    train_ds = ESC50Dataset(ESC50_DIR, meta, split="train")
    test_ds = ESC50Dataset(ESC50_DIR, meta, split="test")
    print(f"Training samples: {len(train_ds)}, Val samples: {len(test_ds)}")

    loader_kwargs = dict(batch_size=args.batch_size, collate_fn=collate,
                         num_workers=args.num_workers, pin_memory=True,
                         persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = AudioCNN(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.002, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1)

    best_accuracy = 0.0
    print("Starting training")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for waveforms, sr, target in progress:
            data = to_spectrogram(waveforms, sr, train=True)
            target = target.to(device)

            optimizer.zero_grad()
            # bfloat16 mixed precision: roughly halves activation memory on a 6 GB GPU
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if np.random.random() > 0.7:
                    data, target_a, target_b, lam = mixup_data(data, target)
                    output = model(data)
                    loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                else:
                    output = model(data)
                    loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_epoch_loss, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for waveforms, sr, target in test_loader:
                data = to_spectrogram(waveforms, sr, train=False)
                target = target.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(data)
                    val_loss += criterion(outputs, target).item()
                correct += (outputs.argmax(1) == target).sum().item()
                total += target.size(0)

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)
        print(f"Epoch {epoch + 1}: train loss {avg_epoch_loss:.4f}, "
              f"val loss {avg_val_loss:.4f}, accuracy {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": accuracy,
                "epoch": epoch,
                "classes": train_ds.classes,
            }, MODEL_DIR / "best_model.pth")
            print(f"New best model saved: {accuracy:.2f}%")

    writer.close()
    print(f"Training completed! Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
