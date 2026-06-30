"""
Generate predictions for test set using 3-fold ensemble (Run 16).
Single-channel grayscale input, native resolution (128×64).
"""
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import CompactSignalCNN
from torchvision import transforms
from PIL import Image
from pathlib import Path

DATA_DIR = Path('../data/raw')
MODEL_DIR = Path('../models')
SUBMISSION_DIR = Path('../data/submissions')
SUBMISSION_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 64)
N_FOLDS = 3


class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['id']
        # Load RGBA, extract G channel
        image = Image.open(img_path)
        arr = np.array(image)[:, :, 1]
        image = Image.fromarray(arr, mode='L')
        if self.transform:
            image = self.transform(image)
        return image, row['id']


def load_dataset_stats() -> tuple[list, list]:
    stats_file = MODEL_DIR / 'dataset_stats.json'
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        print(f'Using dataset stats: mean={stats["mean"]}, std={stats["std"]}')
        return stats['mean'], stats['std']
    print('WARNING: dataset_stats.json not found.')
    return [0.5], [0.5]


def predict():
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    mean, std = load_dataset_stats()

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_dataset = TestDataset(test_df, DATA_DIR / 'test', transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load fold models
    fold_paths = [MODEL_DIR / f'best_model_fold_{i}.pth' for i in range(N_FOLDS)]
    available_folds = [p for p in fold_paths if p.exists()]

    if available_folds:
        model_paths = available_folds
        print(f'Ensemble mode: loading {len(model_paths)} fold models.')
    else:
        print('ERROR: No fold models found.')
        return

    all_probs = None
    ids = None

    for model_path in model_paths:
        print(f'  Loading {model_path.name}...')
        model = CompactSignalCNN(num_classes=5, dropout_p=0.0)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        batch_probs = []
        batch_ids = []

        with torch.no_grad():
            for images, img_ids in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs.cpu())
                if ids is None:
                    batch_ids.extend(img_ids)

        model_probs = torch.cat(batch_probs, dim=0)
        if all_probs is None:
            all_probs = model_probs
            ids = batch_ids
        else:
            all_probs += model_probs

    avg_probs = all_probs / len(model_paths)
    predictions = avg_probs.argmax(dim=1).numpy() + 1  # Back to 1-indexed

    submission = pd.DataFrame({'id': ids, 'label': predictions})
    out_path = SUBMISSION_DIR / 'submission.csv'
    submission.to_csv(out_path, index=False)
    print(f'\nSubmission saved to {out_path}')
    print(f'Label distribution:\n{submission["label"].value_counts().sort_index()}')


if __name__ == '__main__':
    predict()
