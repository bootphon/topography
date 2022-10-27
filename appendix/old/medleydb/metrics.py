"""Provide multiple evaluation metrics for classification on audio or
image data.
"""
import abc
from typing import List

import mir_eval
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from topography.utils.externals import deepsalience

MirEvalInput = List[np.ndarray]


class Metric(abc.ABC):
    """Abstract class to wrap all kind of metrics."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Metric name"""

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def __call__(self, output: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute metric"""


class Accuracy(Metric):
    """Classification accuracy."""

    @property
    def name(self) -> str:
        return "acc"

    def __call__(self, output: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute batch accuracy.

        Parameters
        ----------
        output : torch.Tensor
            Raw outputs of the network.
        labels : torch.Tensor
            Target labels.

        Returns
        -------
        float
            Accuracy on the given batch.
        """
        _, predicted = torch.max(output.data, 1)
        return float((predicted == labels).sum()) / float(output.size(0))


class PitchScore(Metric):
    """Pitch score."""

    freq_grid: np.ndarray = deepsalience.get_freq_grid()

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold
        self._key = "Accuracy"

    @property
    def name(self) -> str:
        return "pitch-acc"

    def _ground_truth_to_mf0(self, labels: np.ndarray) -> List[MirEvalInput]:
        """Convert ground truth pitch labels to mf0 compatible with
        mir_eval.

        Parameters
        ----------
        labels : np.ndarray
            Target labels.

        Returns
        -------
        List[MirEvalInput]
            _description_
        """
        batch_size, patch_size, _ = labels.shape
        idx = np.where(labels == 1)
        est_freqs = [[[] for _ in range(patch_size)] for _ in range(batch_size)]
        for batch, time, freq in zip(idx[0], idx[1], idx[2]):
            est_freqs[batch][time].append(self.freq_grid[freq])
        return [list(map(np.array, ref)) for ref in est_freqs]

    def _pitch_activations_to_mf0(
        self, output: np.ndarray
    ) -> List[MirEvalInput]:
        """Convert output of the network to mf0.
        Adapted from https://github.com/rabitt/ismir2017-deepsalience/blob/4b2ff4449ee9d4b27b9a116d80a9393f246879a7/deepsalience/evaluate.py#L246

        Parameters
        ----------
        output : np.ndarray
            Raw outputs of the network.

        Returns
        -------
        List[MirEvalInput]
            _description_
        """
        batch_size, patch_size, freq_bins = output.shape
        peak_thresh_mat = np.zeros((batch_size, patch_size, freq_bins))
        peaks = signal.argrelmax(output, axis=2)
        peak_thresh_mat[peaks] = output[peaks]

        idx = np.where(peak_thresh_mat >= self.threshold)
        est_freqs = [[[] for _ in range(patch_size)] for _ in range(batch_size)]
        for batch, time, freq in zip(idx[0], idx[1], idx[2]):
            est_freqs[batch][time].append(self.freq_grid[freq])
        return [list(map(np.array, est)) for est in est_freqs]

    def __call__(self, output: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute pitch accuracy using mir_eval.

        Parameters
        ----------
        output : torch.Tensor
            Raw outputs of the network.
        labels : torch.Tensor
            Target labels, of shape ().

        Returns
        -------
        float
            Batch accuracy.
        """
        time = deepsalience.get_time_grid(output.shape[1])
        ref_freqs = self._ground_truth_to_mf0(labels.cpu().detach().numpy())
        est_freqs = self._pitch_activations_to_mf0(
            F.sigmoid(output.cpu().detach()).numpy()
        )
        scores = np.array(
            [
                mir_eval.multipitch.evaluate(time, ref, time, est)[self._key]
                for ref, est in zip(ref_freqs, est_freqs)
            ]
        )
        return scores.mean()
