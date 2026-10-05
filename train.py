from pathlib import Path
import sys
import urllib.request
import zipfile
import pandas as pd
import numpy as np

import soundfile as sf  # LOCAL CHANGE: used instead of torchaudio.load (needs TorchCodec + FFmpeg on Windows)
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import AudioCNN

# LOCAL CHANGE: instead of the Modal image + volumes, ESC-50 is downloaded into ./data
# and the model is saved into ./models
DATA_DIR = Path("data")
MODEL_DIR = Path("models")


def download_esc50():
    # Same steps as the Modal image's wget + unzip commands
    if (DATA_DIR / "ESC-50-master").exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "esc50.zip"
    print("Downloading ESC-50...")
    urllib.request.urlretrieve(
        "https://github.com/karolpiczak/ESC-50/archive/master.zip", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(
            self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        # LOCAL CHANGE: soundfile instead of torchaudio.load. Transposed so the
        # shape is (channels, samples), exactly what torchaudio.load returns.
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label']


def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'{MODEL_DIR}/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)

    download_esc50()
    esc50_dir = DATA_DIR / "ESC-50-master"

    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )

    train_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform)

    val_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="test", transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
    )

    best_accuracy = 0.0

    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            # LOCAL CHANGE: mixed precision (bfloat16) so the model fits in a 6 GB GPU.
            # Without it, memory spills into system RAM and training is ~13x slower.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                if np.random.random() > 0.7:
                    data, target_a, target_b, lam = mixup_data(data, target)
                    output = model(data)
                    loss = mixup_criterion(
                        criterion, output, target_a, target_b, lam)
                else:
                    output = model(data)
                    loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar(
            'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation after each epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                # LOCAL CHANGE: mixed precision, same as in training
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(data)
                    loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(
            f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, f'{MODEL_DIR}/best_model.pth')
            print(f'New best model saved: {accuracy:.2f}%')

    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')


# LOCAL CHANGE: replaces @app.local_entrypoint() + train.remote()
if __name__ == "__main__":
    train()
