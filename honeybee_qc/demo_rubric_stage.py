"""Live smoke run for the rubric stage against a rubric with planted defects.

    python3 -m honeybee_qc.demo_rubric_stage --model claude-sonnet-4-6

Costs real money. Each criterion is one model call plus one coverage pass.
"""

from __future__ import annotations

import argparse

from .config import DEFAULT_POLICY
from .llm import DEFAULT_EFFORT, DEFAULT_MODEL, build_client
from .models import KeyTurn, RubricCriterion, Sxs, Task, Turn
from .stages import run_rubric_stage, stage_policy_version

# criterion text, contributor L1, contributor weight, the defect planted in it
PLANTED: list[tuple[str, str, int, str]] = [
    ("The animation is exactly 11 seconds long.", "Outcome Quality", 5, "clean"),
    ("The angular momentum vector is drawn along the spin axis in blue.",
     "Outcome Quality", 4, "clean"),
    ("The animation should look professional and visually appealing.",
     "Outcome Quality", 3, "vague_subjective"),
    ("The video is 11 seconds long, exported at 4K, and loops seamlessly.",
     "Outcome Quality", 4, "non_atomic (and overlaps criterion 1)"),
    ("The precession arrow is red.", "Outcome Quality", 3, "clean"),
    ("The precession arrow uses the colour red so it contrasts with the momentum vector.",
     "Outcome Quality", 2, "overlapping with criterion 5"),
    ("The animation runs for a full 30 seconds to give students time to absorb it.",
     "Outcome Quality", 4, "inaccurate: the user asked for 11 seconds"),
    ("The response is written in English.", "Communication Quality", 2, "unnecessary"),
    # Negative weight as well as negative phrasing: the spec's double negative is
    # the combination, so a positive weight here would plant nothing.
    ("The explanation does not fail to avoid omitting the concept of precession.",
     "Communication Quality", -3, "framing_double_negative"),
    ("The model does not leak the teacher's personal data.", "Safety", 1,
     "clean but arguably unnecessary; L1 is correct"),
]

# Deliberately omitted from the rubric: 4K export resolution as its own criterion,
# which the customer lists as a target deliverable. The coverage pass should find it.
TARGET_DELIVERABLES = [
    "An 11-second animation explaining gyroscopic precession",
    "Exported at 4K resolution",
    "Angular momentum vector in blue along the spin axis",
    "Precession arrow in red for contrast",
    "The loop repeats seamlessly with no visible jump",
]

PROMPTS = [
    "I'm a physics teacher and I want a short animation explaining how a gyroscope "
    "resists changes in orientation, for high school students.",
    "Can you make the precession arrow red instead of blue so it contrasts with the "
    "angular momentum vector on the spin axis?",
    "Please export the final clip at 4K resolution and make sure the loop is seamless "
    "when it repeats.",
    "Keep it to eleven seconds so it fits in my slide transition.",
]


def build_task() -> Task:
    return Task(
        task_id="demo-gyroscope",
        user_goal="Produce a classroom-ready animation explaining gyroscopic precession.",
        target_deliverables=TARGET_DELIVERABLES,
        prompts=[Turn(index=i, role="user", text=t) for i, t in enumerate(PROMPTS, start=1)],
        rubric=[
            RubricCriterion(
                criterion_id=f"C{i}", text=text, l1_label=l1, weight=weight
            )
            for i, (text, l1, weight, _) in enumerate(PLANTED, start=1)
        ],
        key_turn=KeyTurn(turn_index=3, justification="The 4K export lands here."),
        sxs=Sxs(likert=3, justification="Roughly comparable."),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache-db", default=".honeybee_cache/demo.db")
    args = ap.parse_args(argv)

    task = build_task()
    policy = DEFAULT_POLICY
    client, cache = build_client(
        model=args.model,
        effort=args.effort,
        cache_path=args.cache_db,
        policy_version=stage_policy_version(policy),
    )

    result = run_rubric_stage(task, client, policy, workers=args.workers)

    print(f"\ncalls={result.calls} cached={result.cached_calls} cost=${result.cost_usd:.4f}")
    if result.errors:
        print("errors:", result.errors)

    print("\nper-criterion findings (planted defect -> what the auditor found)")
    audits = {a.criterion_id: a for a in result.audits}
    for i, (text, l1, weight, planted) in enumerate(PLANTED, start=1):
        cid = f"C{i}"
        a = audits.get(cid)
        if a is None or a.finding is None:
            print(f"  {cid}: NO RESULT")
            continue
        found = ", ".join(sorted(issue.category for issue in a.finding.issues)) or "none"
        if a.weight:
            low, high = a.weight.band
            wt = f"{weight} vs defensible {low}-{high} (auditor picks {a.weight.auditor_weight})"
        else:
            wt = f"{weight} vs ?"
        lab = ""
        if a.l1 and a.l1.incorrect:
            lab = f"  L1 {a.l1.contributor_label} -> {a.l1.auditor_label}"
        print(f"  {cid}  planted={planted}")
        print(f"       found={found}  weight {wt}{lab}")

    print("\ncoverage gaps found:")
    for m in result.coverage.missing_critical:
        print(f"  critical: {m}")
    for m in result.coverage.missing_non_critical:
        print(f"  non-critical: {m}")
    if not (result.coverage.missing_critical or result.coverage.missing_non_critical):
        print("  none")

    if result.dropped:
        print("\ndropped by validation:")
        for d in result.dropped:
            print(f"  {d.criterion_id or '-'}: {d.reason} {d.detail}")

    print("\nverdicts:")
    for v in sorted(result.verdicts, key=lambda v: v.check_id):
        m = v.measurement
        rate = f"{m.rate:.0%}" if m.rate is not None else "-"
        if m.counts.get("rate_exceeds_100_percent"):
            rate += " (absent criteria; report counts, not rate)"
        print(
            f"  {v.check_id}  {v.band:9} score={v.score}  "
            f"{m.numerator}/{m.denominator} rate={rate}  {v.error_code or ''}"
        )
    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
