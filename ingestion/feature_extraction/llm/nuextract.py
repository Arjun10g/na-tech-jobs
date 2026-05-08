"""NuExtract-tiny structured-extraction wrapper. Stubbed for Step 1a.

Step 1b wires this to ``numind/NuExtract-tiny-v1.5`` running on CPU. Until then,
``run`` returns an empty dict so the cascade transparently falls back to
"feature unknown" when regex fails.

The interface is intentionally minimal: take description + missing-field list,
return ``{feature_name: Extraction(...)}`` for whatever it can fill in.
"""

from __future__ import annotations

import logging

from ingestion.feature_extraction.confidence import Extraction

logger = logging.getLogger("feature_extraction.nuextract")


class NuExtractStub:
    """Phase 1a stub. Lazy-loaded model lands in Phase 1b."""

    def __init__(self) -> None:
        self.loaded = False

    def run(self, text: str, title: str, missing_fields: list[str]) -> dict[str, Extraction]:
        # Step 1b will load `numind/NuExtract-tiny-v1.5` here and run
        # structured extraction over the missing fields. For Step 1a we
        # log and return nothing — the cascade survives this gracefully.
        if missing_fields:
            logger.debug("NuExtract stub: would request %d missing fields", len(missing_fields))
        return {}
