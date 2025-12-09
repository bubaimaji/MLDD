import os
import numpy as np

# Path to your features (change if needed)
feature_dir = "Itali/features_npy/seg_eGeMAPSv02_Functionals"
FOLDS = 5

# Load all speakers per fold
fold_speakers = []
for i in range(1, FOLDS + 1):
    spk = np.load(os.path.join(feature_dir, f"fold{i}_speaker.npy"), allow_pickle=True)
    fold_speakers.append(set(spk))
    print(f"Fold {i}: {len(spk)} samples from {len(set(spk))} unique speakers")

# Check for overlap
print("\n=== Checking speaker overlap across folds ===")
any_overlap = False
for i in range(FOLDS):
    for j in range(i + 1, FOLDS):
        overlap = fold_speakers[i].intersection(fold_speakers[j])
        if overlap:
            any_overlap = True
            print(f"⚠️  Overlap between fold {i+1} and fold {j+1}: {len(overlap)} speakers -> {list(overlap)[:5]}")
if not any_overlap:
    print(" No speaker overlap across folds — perfectly speaker-independent setup.")
