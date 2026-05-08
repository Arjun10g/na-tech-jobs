"""Per-ATS-provider extractors. Each subclasses `Extractor` from `base`."""

from ingestion.extractors.ashby import AshbyExtractor
from ingestion.extractors.base import Extractor
from ingestion.extractors.greenhouse import GreenhouseExtractor
from ingestion.extractors.lever import LeverExtractor

EXTRACTORS: dict[str, type[Extractor]] = {
    "greenhouse": GreenhouseExtractor,
    "lever": LeverExtractor,
    "ashby": AshbyExtractor,
}

__all__ = [
    "EXTRACTORS",
    "Extractor",
    "GreenhouseExtractor",
    "LeverExtractor",
    "AshbyExtractor",
]
