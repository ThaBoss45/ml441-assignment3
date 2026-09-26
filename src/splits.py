
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    number: int
    fit_end: int
    stop_end: int
    valid_end: int

    @property
    def fit(self) -> range:
        return range(0, self.fit_end)

    @property
    def stop(self) -> range:
        return range(self.fit_end, self.stop_end)

    @property
    def valid(self) -> range:
        return range(self.stop_end, self.valid_end)


def boundaries(n: int) -> tuple[int, int, list[Fold]]:
    development_end = int(0.8 * n)
    block = int(0.1 * n)
    if block < 1:
        raise ValueError("A series is too short for three validation blocks")
    folds = []
    for number in range(1, 4):
        train_end = development_end - (4 - number) * block
        stop_size = max(1, int(0.1 * train_end))
        folds.append(Fold(number, train_end - stop_size, train_end,
                          train_end + block))
    assert folds[0].valid_end == folds[1].stop_end
    assert folds[1].valid_end == folds[2].stop_end
    assert folds[2].valid_end == development_end
    return development_end, block, folds
