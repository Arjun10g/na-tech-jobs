"""Skill extraction via NuExtract-tiny zero-shot structured prompting.

Per CLAUDE.md §7 this is the one classifier that is **not** fine-tuned —
NuExtract-tiny-v1.5 already excels at zero-shot structured extraction, and
we just need a stable canonical taxonomy for the output. The hard work was
done in Phase 2 Step 1b (``ingestion/feature_extraction/llm/nuextract.py``);
this module exposes a focused skills-only API and a canonical-taxonomy
post-processor.
"""

from models.skills.predict import SKILL_TAXONOMY, SkillExtractor

__all__ = ["SKILL_TAXONOMY", "SkillExtractor"]
