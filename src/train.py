"""
Training script for Signal Spectrogram Classification — Run 16.
Model: CompactSignalCNN (~900K params, single-channel grayscale input).
Strategy: Native resolution (128×64), G-channel extraction, SpecAugment, 3-Fold CV.
"""
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import CompactSignalCNN
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path('../data/raw')
MODEL_DIR = Path('../models')
MODEL_DIR.mkdir(exist_ok=True)

# Global Configuration
IMG_SIZE = (128, 64)  # Native height, padded width (55→64)
BATCH_SIZE = 128      # Small images → big batches


class SignalDataset(Dataset):
    """Loads spectrogram images as single-channel grayscale (G channel of viridis)."""
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['id']

        # Load RGBA, extract G channel (highest dynamic range in viridis colormap)
        image = Image.open(img_path)
        arr = np.array(image)[:, :, 1]  # G channel: uint8, shape (128, 55)
        image = Image.fromarray(arr, mode='L')  # Single-channel grayscale

        if self.transform:
            image = self.transform(image)

        label = row['label'] - 1  # Convert to 0-indexed
        return image, label


def compute_dataset_stats(img_dir: Path, csv_path: Path) -> tuple[list, list]:
    """Compute single-channel mean and std over the training set (cached)."""
    stats_file = MODEL_DIR / 'dataset_stats.json'
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        if stats.get('channels') == 1:
            print(f'Loaded cached dataset stats: mean={stats["mean"]}, std={stats["std"]}')
            return stats['mean'], stats['std']
        print('Cached stats are for different channel count. Recomputing...')

    print('Computing single-channel dataset statistics...')
    to_tensor = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    df = pd.read_csv(csv_path)
    pixel_sum = torch.zeros(1)
    pixel_sq_sum = torch.zeros(1)
    pixel_count = 0

    for _, row in df.iterrows():
        img = Image.open(img_dir / row['id'])
        arr = np.array(img)[:, :, 1]  # G channel
        img_gray = Image.fromarray(arr, mode='L')
        t = to_tensor(img_gray)  # [1, H, W]
        pixels = t.view(1, -1)
        pixel_sum += pixels.sum(dim=1)
        pixel_sq_sum += (pixels ** 2).sum(dim=1)
        pixel_count += pixels.shape[1]

    mean = (pixel_sum / pixel_count).tolist()
    std = ((pixel_sq_sum / pixel_count) - (pixel_sum / pixel_count) ** 2).sqrt().tolist()

    stats = {'mean': mean, 'std': std, 'channels': 1}
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Saved dataset stats: mean={mean}, std={std}')
    return mean, std


class FrequencyMasking(nn.Module):
    """SpecAugment: randomly masks a band of frequency rows."""
    def __init__(self, freq_mask_param: int = 20):
        super().__init__()
        self.F = freq_mask_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, H, _ = x.shape
        f = torch.randint(0, self.F + 1, (1,)).item()
        f0 = torch.randint(0, max(1, H - f), (1,)).item()
        x[:, f0:f0 + f, :] = 0.0
        return x


class TimeMasking(nn.Module):
    """SpecAugment: randomly masks a band of time columns."""
    def __init__(self, time_mask_param: int = 10):
        super().__init__()
        self.T = time_mask_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, W = x.shape
        t = torch.randint(0, self.T + 1, (1,)).item()
        t0 = torch.randint(0, max(1, W - t), (1,)).item()
        x[:, :, t0:t0 + t] = 0.0
        return x


def get_transforms(mean: list, std: list, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            FrequencyMasking(freq_mask_param=20),
            TimeMasking(time_mask_param=10),
        ])
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def mixup_data(x, y, alpha=0.3, device='cpu'):
    """Returns mixed inputs, pairs of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_model():
    import sys

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    mean, std = compute_dataset_stats(DATA_DIR / 'train', DATA_DIR / 'train.csv')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(skf.split(train_df, train_df['label']))

    num_epochs = 40
    early_stop_patience = 12
    mixup_warmup_epochs = 5

    fold_accuracies = []
    resume = '--resume' in sys.argv

    for fold in range(3):
        print(f'\n==================== FOLD {fold+1}/3 ====================')
        train_idx, val_idx = folds[fold]
        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]

        train_dataset = SignalDataset(fold_train_df, DATA_DIR / 'train', get_transforms(mean, std, train=True))
        val_dataset = SignalDataset(fold_val_df, DATA_DIR / 'train', get_transforms(mean, std, train=False))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = CompactSignalCNN(num_classes=5, dropout_p=0.5)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

        start_epoch = 0
        best_acc = 0.0
        epochs_without_improvement = 0
        checkpoint_path = MODEL_DIR / f'checkpoint_fold_{fold}.pth'

        if checkpoint_path.exists() and resume:
            print(f'Loading checkpoint for Fold {fold+1}...')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            epochs_without_improvement = checkpoint['epochs_without_improvement']
            print(f'Resumed from Epoch {start_epoch+1} with best_acc={best_acc:.2f}%')

        if resume and start_epoch >= num_epochs:
            print(f'Fold {fold+1} already fully trained. Best Accuracy: {best_acc:.2f}%')
            fold_accuracies.append(best_acc)
            continue

        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_correct = 0
            n_batches = len(train_loader)

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                if epoch >= mixup_warmup_epochs:
                    images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.3, device=device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    train_correct += (lam * (predicted == targets_a).float() + (1 - lam) * (predicted == targets_b).float()).sum().item()
                else:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    train_correct += (predicted == labels).sum().item()

                if (i + 1) % 20 == 0 or (i + 1) == n_batches:
                    print(f'  batch {i+1}/{n_batches}', flush=True)

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100 * val_correct / val_total
            train_acc = 100 * train_correct / len(fold_train_df)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Fold {fold+1} - Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%  LR: {current_lr:.2e}', flush=True)

            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), MODEL_DIR / f'best_model_fold_{fold}.pth')
                print(f'  ✓ New best model saved for Fold {fold+1}: {val_acc:.2f}%', flush=True)
            else:
                epochs_without_improvement += 1

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'epochs_without_improvement': epochs_without_improvement,
            }, checkpoint_path)

            if epochs_without_improvement >= early_stop_patience:
                print(f'Fold {fold+1} early stopping: no improvement for {early_stop_patience} epochs.', flush=True)
                break

        print(f'Fold {fold+1} finished. Best Validation Accuracy: {best_acc:.2f}%')
        fold_accuracies.append(best_acc)

    print('\n==================== CROSS-VALIDATION SUMMARY ====================')
    for fold, acc in enumerate(fold_accuracies):
        print(f'Fold {fold+1}: {acc:.2f}%')
    print(f'Mean CV Accuracy: {np.mean(fold_accuracies):.2f}%')


if __name__ == '__main__':
    train_model()
