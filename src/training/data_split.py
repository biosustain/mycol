"""Single source of truth for the Cellpose train/test split."""

import math

from sklearn.model_selection import train_test_split


def held_out_count(n, test_size):
    """How many of `n` items are held out: sklearn's ceil, clamped so neither side is empty.

    Without the clamp, an extreme split on a small dataset (e.g. 90% test of 5 images)
    leaves no training data and sklearn raises.
    """
    if n < 2:
        return 0
    return min(max(math.ceil(n * test_size), 1), n - 1)


def split_train_test(n, test_size=0.2, seed=42):
    """(train_idx, test_idx); test items are never used for training or tuning."""
    if n < 2:
        return list(range(n)), []
    # absolute count, so the clamp above is what sklearn actually applies
    return train_test_split(
        list(range(n)), test_size=held_out_count(n, test_size), random_state=seed
    )
