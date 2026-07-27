"""Single source of truth for the Cellpose train/test split."""

from sklearn.model_selection import train_test_split


def split_train_test(n, test_size=0.2, seed=42):
    """(train_idx, test_idx); test items are never used for training or tuning."""
    if n < 2:
        return list(range(n)), []
    return train_test_split(list(range(n)), test_size=test_size, random_state=seed)
