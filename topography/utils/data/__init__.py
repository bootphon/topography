"""Wrapped datasets. Can directly be used."""
from topography.utils.data.bird_dcase import BirdDCASE
from topography.utils.data.common import evaluate_with_crop
from topography.utils.data.speechcommands import SpeechCommands

__all__ = ["BirdDCASE", "evaluate_with_crop", "SpeechCommands"]
