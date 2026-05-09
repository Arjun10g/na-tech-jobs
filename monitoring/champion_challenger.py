"""Champion/challenger gating for monthly retraining.

Per CLAUDE.md §7 a new model is promoted only if it **beats production
by ≥1% on the primary metric AND doesn't regress >2% on any secondary
metric**. This module implements that gate so the retrain workflow has
a single point-of-truth for the decision.

Inputs:

- ``challenger`` — the freshly trained model's training_summary.json
  (each ``models/<name>/train.py`` already emits one).
- ``champion`` — the currently-promoted model's training_summary.json,
  pulled from the HF Model repo or fed in from disk.

Outputs:

- A ``GatingDecision`` dataclass with ``promote: bool``, the deltas, and
  a human-readable reason.

Per-model wiring lives in the retrain workflow. This module is
deliberately model-agnostic — it doesn't know whether you're comparing
two seniority classifiers, two role-family classifiers, or two salary
regressors. Caller passes ``primary`` and ``secondary`` metric names.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("monitoring.champion_challenger")

DEFAULT_PRIMARY_LIFT_THRESHOLD: float = 0.01  # +1% on primary
DEFAULT_SECONDARY_REGRESSION_TOLERANCE: float = 0.02  # -2% allowed


# ── Per-model defaults ────────────────────────────────────────────────────


# What each model's training_summary.json calls "the metric we care about".
# Higher = better for all of these.
MODEL_PRIMARY_METRICS: dict[str, str] = {
    "seniority": "eval.eval_f1_macro",
    "role_family": "eval.eval_f1_macro",
}

MODEL_SECONDARY_METRICS: dict[str, list[str]] = {
    "seniority": ["eval.eval_accuracy", "eval.eval_f1_weighted"],
    "role_family": ["eval.eval_accuracy", "eval.eval_f1_weighted"],
}

# Salary is loss-based (lower MAE = better) — invert in code via
# ``higher_is_better=False``. Currently ships separately; this module
# handles both directions.


# ── Helpers ───────────────────────────────────────────────────────────────


def _get_metric(summary: dict[str, Any], dotted_path: str) -> float | None:
    """``"eval.eval_f1_macro"`` → ``summary["eval"]["eval_f1_macro"]``.
    Returns None if the path doesn't resolve."""
    cur: Any = summary
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


@dataclass
class MetricCompare:
    name: str
    champion: float | None
    challenger: float | None
    delta: float | None  # challenger - champion (in metric units)
    delta_pct: float | None  # (challenger - champion) / |champion|


@dataclass
class GatingDecision:
    """The output of the gate."""

    model: str
    promote: bool
    reason: str
    primary: MetricCompare
    secondary: list[MetricCompare]


def _compare(name: str, champ: dict, chal: dict) -> MetricCompare:
    a = _get_metric(champ, name)
    b = _get_metric(chal, name)
    if a is None or b is None:
        return MetricCompare(name=name, champion=a, challenger=b, delta=None, delta_pct=None)
    delta = b - a
    pct = delta / abs(a) if a != 0 else None
    return MetricCompare(name=name, champion=a, challenger=b, delta=delta, delta_pct=pct)


def gate(
    *,
    model: str,
    champion: dict,
    challenger: dict,
    primary_metric: str,
    secondary_metrics: list[str] | None = None,
    higher_is_better: bool = True,
    primary_lift_threshold: float = DEFAULT_PRIMARY_LIFT_THRESHOLD,
    secondary_regression_tolerance: float = DEFAULT_SECONDARY_REGRESSION_TOLERANCE,
) -> GatingDecision:
    """Apply the promotion rule. ``higher_is_better=False`` flips the sign
    so the rule works for loss-style metrics (e.g. salary MAE).

    Promotion rule:

    1. **Primary** must improve by ≥ ``primary_lift_threshold`` (relative).
       For loss metrics (lower=better), challenger MAE must be at least
       ``primary_lift_threshold`` *lower* than champion's.
    2. **No secondary metric** may regress by more than
       ``secondary_regression_tolerance`` (relative).
    3. If the challenger or champion is missing the primary metric
       entirely, **promote=False** with reason logged.
    """
    secondary_metrics = secondary_metrics or []

    primary_cmp = _compare(primary_metric, champion, challenger)
    secondary_cmps = [_compare(m, champion, challenger) for m in secondary_metrics]

    # Bail if we can't even compare.
    if primary_cmp.champion is None or primary_cmp.challenger is None:
        return GatingDecision(
            model=model,
            promote=False,
            reason=f"primary metric {primary_metric!r} missing from champion or challenger",
            primary=primary_cmp,
            secondary=secondary_cmps,
        )

    # Direction-aware lift.
    primary_pct = primary_cmp.delta_pct or 0.0
    if not higher_is_better:
        primary_pct = -primary_pct  # flip so positive == "challenger better"
    primary_pass = primary_pct >= primary_lift_threshold

    if not primary_pass:
        return GatingDecision(
            model=model,
            promote=False,
            reason=(
                f"primary {primary_metric}: lift {primary_pct * 100:+.2f}% "
                f"< required +{primary_lift_threshold * 100:.1f}%"
            ),
            primary=primary_cmp,
            secondary=secondary_cmps,
        )

    # Secondary regression check.
    regressed: list[MetricCompare] = []
    for s in secondary_cmps:
        if s.delta_pct is None:
            continue
        s_pct = s.delta_pct if higher_is_better else -s.delta_pct
        if s_pct < -secondary_regression_tolerance:
            regressed.append(s)
    if regressed:
        names = ", ".join(f"{r.name}({(r.delta_pct or 0) * 100:+.2f}%)" for r in regressed)
        return GatingDecision(
            model=model,
            promote=False,
            reason=f"secondary regression(s) > {secondary_regression_tolerance * 100:.1f}%: {names}",
            primary=primary_cmp,
            secondary=secondary_cmps,
        )

    return GatingDecision(
        model=model,
        promote=True,
        reason=(
            f"primary {primary_metric}: +{primary_pct * 100:.2f}% "
            f"(threshold +{primary_lift_threshold * 100:.1f}%); "
            f"no secondary regression > {secondary_regression_tolerance * 100:.1f}%"
        ),
        primary=primary_cmp,
        secondary=secondary_cmps,
    )


def gate_classifier(
    model_name: str,
    champion: dict,
    challenger: dict,
    **kwargs: Any,
) -> GatingDecision:
    """Convenience: pre-fill primary/secondary for the seniority +
    role_family classifiers."""
    primary = MODEL_PRIMARY_METRICS.get(model_name, "eval.eval_f1_macro")
    secondary = MODEL_SECONDARY_METRICS.get(model_name, [])
    return gate(
        model=model_name,
        champion=champion,
        challenger=challenger,
        primary_metric=primary,
        secondary_metrics=secondary,
        higher_is_better=True,
        **kwargs,
    )


def gate_salary_regressor(
    champion: dict,
    challenger: dict,
    **kwargs: Any,
) -> GatingDecision:
    """Convenience: salary regressor uses MAE (lower is better)."""
    return gate(
        model="salary",
        champion=champion,
        challenger=challenger,
        primary_metric="test_mae",  # tier-5 XGBoost output structure
        secondary_metrics=["cv_mae", "test_mape"],
        higher_is_better=False,
        **kwargs,
    )


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Model name (seniority / role_family / salary)")
    p.add_argument("--champion", required=True, help="Path to champion training_summary.json")
    p.add_argument("--challenger", required=True, help="Path to challenger training_summary.json")
    p.add_argument(
        "--out-path",
        default=None,
        help="Where to write the decision JSON (defaults to stdout only)",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _decision_to_dict(d: GatingDecision) -> dict:
    return {
        "model": d.model,
        "promote": d.promote,
        "reason": d.reason,
        "primary": d.primary.__dict__,
        "secondary": [s.__dict__ for s in d.secondary],
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    champion = json.loads(Path(args.champion).read_text())
    challenger = json.loads(Path(args.challenger).read_text())

    if args.model in ("seniority", "role_family"):
        decision = gate_classifier(args.model, champion, challenger)
    elif args.model == "salary":
        decision = gate_salary_regressor(champion, challenger)
    else:
        raise SystemExit(f"unknown model: {args.model}")

    payload = _decision_to_dict(decision)
    print(json.dumps(payload, indent=2, default=str))
    if args.out_path:
        Path(args.out_path).write_text(json.dumps(payload, indent=2, default=str))
    return 0 if decision.promote else 2  # 2 = "do not promote"


if __name__ == "__main__":
    raise SystemExit(main())
