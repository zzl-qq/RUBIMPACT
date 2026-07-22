"""Streaming HDF5 output (framework S9.3)."""
import h5py
import numpy as np
from pathlib import Path


class HDF5Writer:
    def __init__(self, output_dir: Path):
        self._dir = Path(output_dir)
        self._files = {}
        self._datasets = {}
        self._counters = {}

    def open_history(self, name: str, row_size: int) -> None:
        path = self._dir / "history" / f"{name}.h5"
        f = h5py.File(path, "w")
        dset = f.create_dataset(name, shape=(0, row_size), maxshape=(None, row_size),
                                dtype="float64", chunks=(1024, row_size),
                                compression="gzip", compression_opts=4)
        self._files[name] = f
        self._datasets[name] = dset
        self._counters[name] = 0

    def append_history(self, name: str, row: np.ndarray) -> None:
        dset = self._datasets[name]
        n = self._counters[name]
        dset.resize((n + 1, dset.shape[1]))
        dset[n, :] = row
        self._counters[name] = n + 1

    def dump_field(self, name: str, step: int, data: np.ndarray) -> None:
        path = self._dir / "field" / f"{name}_step{step:08d}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(name, data=data, compression="gzip", compression_opts=1)

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._datasets.clear()
        self._counters.clear()
