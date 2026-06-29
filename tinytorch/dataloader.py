import random
from tinytorch.ndarray import NDArray
from tinytorch.tensor import Tensor

import math 

class Dataset:

    def __init__(self, X: list, y: list):
        self.X = Tensor(NDArray((len(X), len(X[0])), data=[v for row in X for v in row]))
        self.y = Tensor(NDArray((len(y),), data=list(map(float, y))))

    def __len__(self):
        return self.y.data.size

    def __getitem__(self, index):
        X_row = self.X.data[index]              # NDArray row, shape (features,)
        y_val = self.y.data._storage[index]     # raw float
        return Tensor(X_row), Tensor(NDArray((1,), data=[y_val]))


class DataLoader:

    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(self.dataset), self.batch_size):
            chunk = indices[start : start + self.batch_size]

            X_rows = []
            y_vals = []
            for i in chunk:
                x, y = self.dataset[i]
                X_rows.append(x.data)
                y_vals.append(y.data)

            features = X_rows[0].shape[0]
            X_batch = NDArray((len(chunk), features), data=[v for row in X_rows for v in row._storage])
            y_batch = NDArray((len(chunk),), data=[arr._storage[0] for arr in y_vals])

            yield Tensor(X_batch), Tensor(y_batch)
