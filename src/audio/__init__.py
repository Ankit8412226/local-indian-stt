"""src/audio package."""
from src.audio.inspector import AudioMetadata, inspect
from src.audio.converter import load_audio
from src.audio.quality import QualityReport, analyze
from src.audio.normalizer import peak_normalize, rms_normalize
from src.audio.processor import ProcessedAudio, process_audio

__all__ = [
    "AudioMetadata",
    "inspect",
    "load_audio",
    "QualityReport",
    "analyze",
    "peak_normalize",
    "rms_normalize",
    "ProcessedAudio",
    "process_audio",
]
