import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
FOLDS = 5

BASE = "/home/bubai-maji/bubai/Itali/features_npy/IS10"

def load_fold(fold):
    y = np.load(f"{BASE}/fold{fold}_y.npy").astype(int)
    spk = np.load(
        f"{BASE}/fold{fold}_speaker.npy",
        allow_pickle=True
    )
    return y, spk

for ratio in [1.0, 0.75, 0.50]:

    print("\n" + "=" * 60)
    print(f"TRAINING DATA = {int(ratio*100)}%")
    print("=" * 60)

    train_spk_counts = []
    test_spk_counts = []

    for fold_idx in range(FOLDS):

        y_list = []
        spk_list = []

        for f in range(1, FOLDS + 1):
            y, spk = load_fold(f)
            y_list.append(y)
            spk_list.append(spk)

        # test fold
        y_test = y_list[fold_idx]
        spk_test = spk_list[fold_idx]

        # train folds
        y_train = np.hstack(
            [y_list[i] for i in range(FOLDS) if i != fold_idx]
        )

        spk_train = np.hstack(
            [spk_list[i] for i in range(FOLDS) if i != fold_idx]
        )

        unique_spk = np.unique(spk_train)

        speaker_labels = []
        for s in unique_spk:
            speaker_labels.append(
                y_train[np.where(spk_train == s)[0][0]]
            )

        speaker_labels = np.array(speaker_labels)

        if ratio < 1.0:
            keep_spk, _ = train_test_split(
                unique_spk,
                train_size=ratio,
                stratify=speaker_labels,
                random_state=SEED
            )
            train_speakers = len(keep_spk)
        else:
            train_speakers = len(unique_spk)

        test_speakers = len(np.unique(spk_test))

        train_spk_counts.append(train_speakers)
        test_spk_counts.append(test_speakers)

        print(
            f"Fold {fold_idx+1}: "
            f"Train Speakers={train_speakers}, "
            f"Test Speakers={test_speakers}"
        )

    print("\nAverage across folds")
    print(
        f"Train Speakers = {np.mean(train_spk_counts):.2f}"
    )
    print(
        f"Test Speakers  = {np.mean(test_spk_counts):.2f}"
    )