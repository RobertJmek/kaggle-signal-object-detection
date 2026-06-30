"""
Upload the signal-object-detection data as a Kaggle dataset (for use in the Kaggle notebook).
Uploads only: train.csv, test.csv, sample_submission.csv, train/, test/  (skips *.zip and .DS_Store).

Auth: uses ~/.kaggle/kaggle.json (present) or KAGGLE_USERNAME / KAGGLE_KEY env vars.
Run:  python3 upload_dataset.py
"""
import kagglehub

# Real data dir (resolved from data/raw symlink)
SRC = '/Users/robertjmek/.cache/kagglehub/competitions/signal-object-detection'

# <username>/<dataset-slug> — change the slug if you want; the notebook auto-detects either way.
HANDLE = 'robertnchis/signal-object-detection-data'

# skip the big redundant zips and OS junk (combined with kagglehub's DEFAULT_IGNORE_PATTERNS)
IGNORE = ['*.zip', '.DS_Store', '**/.DS_Store']

print(f'Uploading {SRC}\n     -> kaggle dataset "{HANDLE}" (ignoring {IGNORE})')
kagglehub.dataset_upload(HANDLE, SRC, ignore_patterns=IGNORE)
print('\nDone. In your Kaggle notebook: right panel -> "+ Add Input" -> Datasets tab ->')
print(f'   search "{HANDLE.split("/")[-1]}" -> Add. Then run the data cell (it finds train.csv recursively).')
