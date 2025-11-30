from datasets import load_dataset
from pathlib import Path
import os

def download_c4_subset():

    data_dir = Path("./data/c4")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Start downloading the C4 dataset...")
    print("=" * 80)
    
    print("\nDownload training set...")
    train_dataset = load_dataset(
        "allenai/c4",
        "en", 
        split="train",
        streaming=True
    )
    
    # first 100,000 data points were used for training (~ 300MB).
    train_samples = []
    num_train_samples = 100000
    
    print(f"Extract {num_train_samples:,} training samples...")
    for i, sample in enumerate(train_dataset):
        if i >= num_train_samples:
            break
        train_samples.append(sample)
        if (i + 1) % 10000 == 0:
            print(f"  Extracted {i+1:,} / {num_train_samples:,} samples")
    
    # save training sets
    from datasets import Dataset
    train_dataset_local = Dataset.from_dict({
        'text': [s['text'] for s in train_samples],
        'timestamp': [s['timestamp'] for s in train_samples],
        'url': [s['url'] for s in train_samples],
    })
    
    train_path = data_dir / "train"
    print(f"\nSave training set to {train_path}...")
    train_dataset_local.save_to_disk(str(train_path))
    print(f"✓ Training set saved successfully.")
    
    # download
    print("\nDownload validation set...")
    val_dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True
    )
    
    # first 10,000 records were used for verification (approximately 30MB).
    val_samples = []
    num_val_samples = 10000
    
    print(f"Extract {num_val_samples:,} validation samples...")
    for i, sample in enumerate(val_dataset):
        if i >= num_val_samples:
            break
        val_samples.append(sample)
        if (i + 1) % 2000 == 0:
            print(f"  Extracted {i+1:,} / {num_val_samples:,} samples")
    
    # save validation set
    val_dataset_local = Dataset.from_dict({
        'text': [s['text'] for s in val_samples],
        'timestamp': [s['timestamp'] for s in val_samples],
        'url': [s['url'] for s in val_samples],
    })
    
    val_path = data_dir / "validation"
    print(f"\nSave validation set to {val_path}...")
    val_dataset_local.save_to_disk(str(val_path))
    print(f"✓ Validation set saved!")
    
    # stats
    print("\n" + "=" * 80)
    print("Download complete")
    print("=" * 80)
    print(f"Traning set: {len(train_samples):,} samples")
    print(f"Validation set: {len(val_samples):,} samples")
    print(f"Saved to: {data_dir.absolute()}")
    
    import json
    train_size = len(json.dumps(train_samples)) / 1024 / 1024
    val_size = len(json.dumps(val_samples)) / 1024 / 1024
    print(f"~{train_size + val_size:.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    download_c4_subset()