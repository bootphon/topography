"""Utilities to create a PyTorch Dataset of pitch annotated tracks from
MedleyDB.
You need to request an access to MedleyDB, download it and set
MEDLEYDB_PATH accordingly beforehand. You also need to preprocess the input
data, with using the script adapted from
https://github.com/rabitt/ismir2017-deepsalience.
This can be done with scripts/medleydb.sh
"""
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

OPTIONS = (
    "bass",
    "melody1",
    "melody2",
    "melody3",
    "multif0_complete",
    "multif0_incomplete",
    "solo_pitch",
    "vocal",
)


class Metadata(NamedTuple):
    """Metadata container."""

    id: str
    input_path: str
    output_path: str
    start: int
    end: int

    @classmethod
    def from_csv(cls, row: List[str]):
        """Create a metadata entry from a CSV row."""
        return cls(row[0], str(row[1]), str(row[2]), int(row[3]), int(row[4]))

    def __str__(self) -> str:
        """String representation, used for CSV export."""
        return ",".join([str(field) for field in self])


def _build_metadata(root: Path, patch_size: int) -> Dict[int, Metadata]:
    """Build and save the metadata CSV.

    Parameters
    ----------
    root : Union[str, Path]
        Path to the root directory.
    patch_size : int
        Patch size to be used.

    Returns
    -------
    Dict[int, Metadata]
        Metadata for all samples.
    """
    idx = 0
    all_metadata = {}

    # Check dataset coherence
    root = Path(root)
    input_path, output_path = root.joinpath("inputs"), root.joinpath("outputs")
    if not root.exists() or not input_path.exists() or not output_path.exists():
        raise ValueError(
            f"MedleyDB dataset at {root} was not build beforehand."
        )
    input_files = sorted(input_path.glob("*.npy"))
    output_files = sorted(output_path.glob("*.npy"))
    if len(input_files) != len(output_files):
        raise ValueError(
            "Not same number of files in inputs and outputs directories."
        )

    for inp, out in zip(input_files, output_files):
        inp_id = inp.name.replace("_input.npy", "")
        if inp_id != out.name.replace("_output.npy", ""):
            raise ValueError(
                f"Input {inp.name} and output {out.name} files do not match."
            )
        hcqt = np.load(inp, mmap_mode="r")
        for start in range(0, hcqt.shape[-1] - patch_size, patch_size):
            all_metadata[idx] = Metadata(
                id=inp_id,
                input_path=str(inp.resolve()),
                output_path=str(out.resolve()),
                start=start,
                end=start + patch_size,
            )
            idx += 1

    # Write metadata
    with open(root.joinpath("metadata.csv"), "w", encoding="utf-8") as file:
        file.write(
            "\n".join([f"{i}," + str(data) for i, data in all_metadata.items()])
        )
    return all_metadata


class MedleyDBDataset(Dataset):
    """Subset of MedleyDB with annotated pitch."""

    def __init__(
        self,
        root: Union[str, Path],
        patch_size: int = 50,
        option: str = "multif0_complete",
    ) -> None:
        """Create the dataset.

        Parameters
        ----------
        root : Union[str, Path]
            Path to the root directory. It is where the usable
            and preprocessed subset of MedleyDB will be located,
            it is different from MEDLEYDB_PATH.
        patch_size : int, optional
            Patch size for the samples, by default 50
        option : bool, optional
            Type of data to use. Must be in:
            ("bass", "melody1", "melody2", "melody3", "multif0_complete",
             "multif0_incomplete", "solo_pitch", "vocal"), by default
             "multif0_complete".
        """
        super().__init__()
        if option not in OPTIONS:
            raise ValueError(f"Invalid option {option}")
        self.root = Path(root).joinpath(option)
        self.patch_size = patch_size
        self._metadata = {}
        self._lazy_loaded = {}

        metadata_path = self.root.joinpath("metadata.csv")
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as file:
                for line in file.read().splitlines():
                    idx, *raw_metadata = line.split(",")
                    metadata = Metadata.from_csv(raw_metadata)
                    self._metadata[int(idx)] = metadata
                    for path in (metadata.input_path, metadata.output_path):
                        if path not in self._lazy_loaded:
                            self._lazy_loaded[path] = np.load(
                                path, mmap_mode="r+"
                            )
        else:
            self._metadata = _build_metadata(self.root, patch_size)
            for metadata in self._metadata:
                if path not in self._lazy_loaded:
                    self._lazy_loaded[path] = np.load(path, mmap_mode="r+")

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Metadata]:
        """Get a sample from the dataset. Thanks to mmap in NumPy,
        only the used part of the HCQT is loaded.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        torch.Tensor
            HCQT, reference pitch and metadata.
        """
        if index >= len(self):
            raise IndexError
        metadata = self._metadata[index]
        start, end = metadata.start, metadata.end
        hcqt = self._lazy_loaded[metadata.input_path][:, :, start:end]
        pitch = self._lazy_loaded[metadata.output_path][:, start:end]
        return (
            torch.from_numpy(hcqt).permute(0, 2, 1),
            torch.from_numpy(pitch).T,
        )

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self._metadata)
