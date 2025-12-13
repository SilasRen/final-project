from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Fold:
    train_idx: list
    val_idx: list
    test_idx: list


def rolling_time_series_folds(
    n: int,
    train_size: int,
    val_size: int,
    test_size: int,
    step: int,
) -> List[Fold]:
    folds: List[Fold] = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        if test_end > n:
            break

        train_idx = list(range(train_start, train_end))
        val_idx = list(range(train_end, val_end))
        test_idx = list(range(val_end, test_end))
        folds.append(Fold(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))

        start += step

    return folds
