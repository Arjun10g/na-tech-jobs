"""NuExtract-tiny structured-extraction wrapper (Phase 2 Step 1b).

Loads ``numind/NuExtract-tiny-v1.5`` once per process. Per-call:
1. Trim description to ``MAX_INPUT_CHARS`` (the model's context window is 8k
   tokens; NA tech postings comfortably fit at 4k chars).
2. Build a dynamic JSON-schema template covering only the missing fields.
3. Run greedy generation, parse the ``<|output|>...<|end-output|>`` block.
4. Coerce primitives + validate enum values, drop anything malformed.

If transformers / torch aren't available (e.g. running CI with only the dev
group), ``run`` returns an empty dict and the cascade gracefully degrades to
Tier 1 only.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ingestion.feature_extraction.confidence import Extraction

logger = logging.getLogger("feature_extraction.nuextract")

MODEL_ID = "numind/NuExtract-tiny-v1.5"
# Description-text limit. NuExtract-tiny has 8k context, but transformer
# compute scales with sequence length (squared for attention) so we keep this
# tight. Most NA tech postings front-load requirements / location / pay
# information; truncating after ~2k chars rarely loses signal for our schema.
MAX_INPUT_CHARS = 2000
# Output JSON for our 6-field schema tops out around 150 tokens.
MAX_NEW_TOKENS = 200

# Each entry is the JSON-schema "type indicator" NuExtract expects in the
# template: "" for a string, [""] for a list of strings, etc. Numbers and
# booleans are also represented as "" (NuExtract returns strings either way;
# we coerce in Python).
LLM_FIELD_SCHEMAS: dict[str, Any] = {
    "min_years_experience": "",
    "min_education": "",
    "requires_security_clearance": "",
    "clearance_level": "",
    "requires_citizenship": [""],
    "offers_visa_sponsorship": "",
    "remote_policy_extracted": "",
    "on_call_required": "",
    "offers_equity": "",
    "bonus_mentioned": "",
    "tech_stack": [""],
    "industry_experience": [""],
    "team_or_department": "",
}

# Valid enum values per field. NuExtract sometimes paraphrases — we drop
# anything outside this set.
ENUM_VALUES: dict[str, set[str]] = {
    "min_education": {"high_school", "associates", "bachelors", "masters", "phd"},
    "clearance_level": {"public_trust", "confidential", "secret", "top_secret", "ts_sci"},
    "offers_visa_sponsorship": {"yes", "no", "unspecified"},
    "remote_policy_extracted": {"onsite", "hybrid", "remote", "remote-na"},
}

INT_FIELDS: set[str] = {"min_years_experience"}
BOOL_FIELDS: set[str] = {
    "requires_security_clearance",
    "on_call_required",
    "offers_equity",
    "bonus_mentioned",
}
LIST_FIELDS: set[str] = {
    "requires_citizenship",
    "tech_stack",
    "industry_experience",
}

# Confidence flat-rate for Tier 2. We could try to use logits but NuExtract's
# greedy decoding makes this finicky; constant 0.62 keeps the value above the
# Tier 1 threshold (0.6) without overpowering high-confidence regex hits.
LLM_CONFIDENCE: float = 0.62


def _parse_bool(s: Any) -> bool | None:
    if isinstance(s, bool):
        return s
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return None


def _parse_int(s: Any) -> int | None:
    if isinstance(s, int):
        return s
    if not isinstance(s, str):
        return None
    m = re.search(r"\d{1,3}", s)
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def _coerce_field(name: str, raw: Any) -> Any:
    """Coerce raw NuExtract output (always strings/lists of strings) into
    typed values. Returns None when the value is empty / invalid."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip() == "":
        return None
    if isinstance(raw, list) and not raw:
        return None
    if isinstance(raw, list) and all(isinstance(x, str) and x.strip() == "" for x in raw):
        return None

    if name in INT_FIELDS:
        return _parse_int(raw)
    if name in BOOL_FIELDS:
        return _parse_bool(raw)
    if name in ENUM_VALUES:
        if not isinstance(raw, str):
            return None
        v = raw.strip().lower().replace(" ", "_").replace("-", "_")
        # Common paraphrase fixups before validating.
        v = {
            "ts_sci": "ts_sci",
            "ts/sci": "ts_sci",
            "topsecret": "top_secret",
            "top_secret_sci": "ts_sci",
            "bachelor": "bachelors",
            "master": "masters",
            "doctorate": "phd",
            "remote_north_america": "remote-na",
            "remote_in_north_america": "remote-na",
        }.get(v, v)
        # remote-na has a hyphen; keep it.
        v = v.replace("remote_na", "remote-na")
        if v in ENUM_VALUES[name]:
            return v
        return None
    if name in LIST_FIELDS:
        if isinstance(raw, list):
            cleaned = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
            return cleaned or None
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        return None
    if isinstance(raw, str):
        v = raw.strip()
        return v or None
    return None


def _build_schema(missing_fields: list[str]) -> dict[str, Any]:
    """Pick out the LLM-eligible subset of ``missing_fields``."""
    return {f: LLM_FIELD_SCHEMAS[f] for f in missing_fields if f in LLM_FIELD_SCHEMAS}


def _build_prompt(text: str, schema: dict[str, Any]) -> str:
    schema_str = json.dumps(schema, indent=4)
    body = text[:MAX_INPUT_CHARS]
    return "<|input|>\n### Template:\n" + schema_str + "\n### Text:\n" + body + "\n<|output|>\n"


_OUTPUT_RE = re.compile(r"<\|output\|>(.*?)(?:<\|end-output\|>|$)", re.DOTALL)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_output(decoded: str) -> dict[str, Any]:
    """Extract the JSON block from NuExtract's output. Defensive against
    truncation / hallucinated trailing text."""
    body_match = _OUTPUT_RE.search(decoded)
    body = body_match.group(1).strip() if body_match else decoded.strip()
    json_match = _JSON_RE.search(body)
    if not json_match:
        return {}
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return {}


class NuExtract:
    """Singleton wrapper around NuExtract-tiny. Lazy-loads on first use."""

    def __init__(self) -> None:
        self.loaded = False
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def _ensure_loaded(self) -> bool:
        if self.loaded:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.warning(
                "transformers/torch not installed; LLM tier disabled. "
                "Install with `uv sync --extra ml`."
            )
            return False

        logger.info("loading %s …", MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        # Causal-LM batching needs left-padding so generation continues from the
        # final non-pad token regardless of input length.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Pick the best available device + dtype. Apple Silicon MPS gives a
        # ~3-5x boost over CPU for this model size; on Linux GPU we'd use cuda.
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32  # mps/cuda half doesn't help on CPU
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.loaded = True
        logger.info("loaded NuExtract on %s (dtype=%s)", device, dtype)
        return True

    def _generate(self, prompt: str) -> str:
        return self._generate_batch([prompt])[0]

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        import torch

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=6000,
            padding=True,
        ).to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Left-padded inputs all share the same prefix length, so slicing at
        # ``inputs["input_ids"].shape[1]:`` keeps only the newly generated
        # tokens per row.
        prefix_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(output_ids[i, prefix_len:], skip_special_tokens=False)
            for i in range(output_ids.shape[0])
        ]

    def run(self, text: str, title: str, missing_fields: list[str]) -> dict[str, Extraction]:
        results = self.run_batch([(text, title, missing_fields)])
        return results[0] if results else {}

    def run_batch(
        self,
        items: list[tuple[str, str, list[str]]],
    ) -> list[dict[str, Extraction]]:
        """Batched variant. Each item is ``(text, title, missing_fields)``.

        Items whose schema is empty (no LLM-eligible missing fields) or whose
        text is blank short-circuit to an empty dict without hitting the model.
        Items that *do* need the model are tokenized + padded together for one
        ``generate()`` call.
        """
        if not items:
            return []

        # Pre-build schemas + prompts; record which items skip the model.
        prompts: list[str] = []
        schemas: list[dict[str, Any]] = []
        result_idx: list[int] = []  # indexes in ``items`` that get LLM
        out: list[dict[str, Extraction]] = [{} for _ in items]

        for i, (text, title, missing) in enumerate(items):
            schema = _build_schema(missing)
            if not schema or not text:
                continue
            body = (f"# {title}\n\n{text}" if title else text)[:MAX_INPUT_CHARS]
            prompts.append(_build_prompt(body, schema))
            schemas.append(schema)
            result_idx.append(i)

        if not prompts:
            return out

        if not self._ensure_loaded():
            return out

        try:
            decoded_batch = self._generate_batch(prompts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NuExtract batch generation failed: %s", exc)
            return out

        # ``result_idx``, ``schemas``, ``decoded_batch`` are all parallel arrays.
        for idx, schema, decoded in zip(result_idx, schemas, decoded_batch, strict=True):
            parsed = _parse_output(decoded)
            if not isinstance(parsed, dict):
                continue
            extractions: dict[str, Extraction] = {}
            for name, raw in parsed.items():
                if name not in schema:
                    continue
                value = _coerce_field(name, raw)
                if value is None:
                    continue
                extractions[name] = Extraction(
                    value=value,
                    confidence=LLM_CONFIDENCE,
                    source="llm",
                    rule_id="nuextract_v1",
                )
            out[idx] = extractions
        return out


# Backwards-compatible alias kept so the cascade's previous import keeps working.
NuExtractStub = NuExtract
