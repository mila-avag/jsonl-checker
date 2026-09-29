"""Paste-safety checks the fixer must pass on every run.

These are the QC miss families that used to be a manual "go look" step:

  1. Produced-outcome checkbox vs files the assistant actually generated.
  2. Dimension turn cites: N/A and a score of 4 or 5 may be empty. Scores 1–3 must
     be in-range model turns that exist in the transcript.
  3. SxS 1–7: only 4 is a tie; 3 and 5 are slight wins. Likert, writeup, and
     responseIdx must agree.
  4. Cited turn text has to be present in the hydrated transcript (the semantic
     "does this turn support the justification claim" pass is the subagent
     checklist at the bottom of this file).
  5. Dimension-score patches must not leave Inconsistent Ranking: if the mean
     of the pasted 1–5s favors the other model from the 1–7 Likert, the SxS
     writeup has to explain the inversion or those score patches are dropped.
  6. Pair slot: writeups must not name Model/Response/Product X when X is not
     in the comparison (Model B on an A-vs-C sheet).
  7. Likert magnitude: 3/5 cannot say "much better"; 1/7 cannot say "slightly".
  8. N/A justifications cannot keep a 1–5 score.
  9. If the justification names turns and the selected list is disjoint, the
     list is wrong.
  10. Copied A/C justifications and turn-1-only cites on a 7+ turn transcript.

`python -m honeybee_qc.fixer_invariants` re-checks a fixer directory against
the original tasks and exits 1 on any hold. The fixer calls the same functions
inside `apply_plan` so a bad paste never ships just because nobody reran this.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from .context import ARTIFACT_PLACEHOLDER_PATTERNS
from .ingest import load_taskattempts_csv
from .models import DimensionRating, ModelSubmission, Sxs, Task
from .preflight import load_tasks
from .projects import ProjectProfile, profile_by_key, profile_for_project

NA_RE = re.compile(
    r"\b(n/?a|not[- ]applicable|no tools?(?: were)? used|did not use (?:any )?tools?|"
    r"tools? were not used|no tool(?:s| calls?)|without tools?)\b",
    re.IGNORECASE,
)
TIE_RE = re.compile(
    r"\b(tie|tied|toss-?up|dead heat|equally (?:good|strong|matched)|no (?:clear )?winner|"
    r"neither|mixed|comparable|roughly even|nearly identical|equivalent|"
    r"indistinguishable|essentially the same)\b",
    re.IGNORECASE,
)
INVERSION_RE = re.compile(
    r"\b(?:despite|even though|although|in spite of|outweigh(?:s|ed)?|"
    r"more (?:important|severe|decisive)|decisive (?:failure|advantage|issue)|"
    r"lower (?:dimension |overall )?scores?|wins? (?:fewer|less) dimensions?|"
    r"dimension scores? (?:favor|favour|lean))\b",
    re.IGNORECASE,
)
TURN_RE = re.compile(r"\b(?:turns?|t)\s*#?\s*(\d+)\b", re.IGNORECASE)
SLOT_MENTION_RE = re.compile(
    r"\b(?:model|response|product)\s+([ABCD])\b", re.IGNORECASE
)
MUCH_BETTER_RE = re.compile(
    r"\b(?:much|clearly|significantly|substantially|far)\s+better\b", re.IGNORECASE
)
SLIGHTLY_BETTER_RE = re.compile(r"\bslightly better\b", re.IGNORECASE)
NA_SCORE_TOKENS = frozenset({"not-applicable", "n/a", "na", "none"})
COPIED_JUST_MIN_LEN = 120
COPIED_JUST_RATIO = 0.85
SLOT_LETTER_RE = re.compile(r"\b(?:model|response|product)\s+[ABCD]\b", re.IGNORECASE)
FILE_EXT_RE = re.compile(
    r"\.(pdf|xlsx?|csv|docx?|pptx?|png|jpe?g|gif|webp|zip|svg|mp4|wav)\b",
    re.IGNORECASE,
)
GENERATE_VERB_RE = re.compile(
    r"\b(created|generated|exported|attached|saved|rendered|produced|downloaded)\b",
    re.IGNORECASE,
)
STRONG_PLACEHOLDERS = frozenset(
    {"media_player", "ready", "here_is", "produced", "rendered_media"}
)

# Likert 1–3 left, 4 tie, 5–7 right. Do not use Policy.likert_neutral=(3,4,5).
LEFT_LIKERT = frozenset({1, 2, 3})
TIE_LIKERT = frozenset({4})
RIGHT_LIKERT = frozenset({5, 6, 7})

RANKING_CONSISTENCY_NOTE = """
Dimension scores and the SxS ranking are one judgment. After any 1–5 change:
  * Likert 1–3 prefers the left model, 4 is the only tie, 5–7 prefers the right.
    3 and 5 are slight wins, not ties.
  * If the mean of the (new) dimension scores favors the other model from the
    Likert, the preference writeup MUST explain the inversion (despite / even
    though / outweigh / lower dimension scores). A one-line "Verdict: C is much
    better" prepend is not an explanation, and neither is "the pairwise
    outcome outweighs the mean of the 1-5s." Name the turns.
    * If you cannot explain the inversion, do not change the dimension scores.
    Unexplained inversions are dropped from the paste sheet.
  * Name only the slots in this pair (A vs C must not say Model B). 3/5 =
    slightly better; 1/7 = much better. Ignore the Verdict scale legend.
"""

SUBAGENT_PROMPT = """You are checking a Honeybee fixer paste sheet against the actual transcripts.

For EVERY task in the fixer output (not a sample), verify all ten families. Use the hydrated conversation, not the contributor's memory of it. Full 24-hex task IDs. Speak in check names.

1. Produced outcome (check 82): If the assistant generated/exported/attached a real file, the produced-outcome checkbox cannot stay No. Prompt-only file talk with no file is not a flip.
2. Dimension turns (check 310): N/A / no-tools / no-memory lists must be []. A dimension scored 4 or 5 may also be empty — do not hold it for missing cites. Scores 1–3 must cite an in-range model turn. The model text at that turn must support the SPECIFIC justification claim, not merely share a topic. User-turn complaints are not evidence. Off-by-one of real model text is OK. Fail the task at ≥3 incorrect/missing. Do not invent turn numbers that are not in the conversation; do pick a real in-range turn when a 1–3 list is empty.
3. SxS 1–7 (checks 400/470): 1 = left much better … 4 = tie … 7 = right much better. ONLY 4 is a tie. 3 and 5 are slight wins. Likert, writeup, and responseIdx (0 / -1 / 1) must agree. Do not flip a winner to the other side.
4. Transcript verification: open every cited turn. If the share page was a stub, hold the task; do not invent cites.
5. Inconsistent Ranking (check 460): After any dimension_rating patch, if the mean of the 1–5s favors the other model from the Likert, the SxS writeup must explain the inversion (despite / even though / outweigh). A prepended "Verdict: C is much better" sentence is not an explanation. Mixed dominance is not a free pass when the averages contradict.
6. Wrong pair slot: an A-vs-C writeup that names Model B / Response B (or any slot not in the pair) is a hold. Overlay verdict prepends do not clean the body.
7. Likert magnitude: 3 and 5 are slightly better; 1 and 7 are much better. Body text that says "much" on a 3/5 (or "slightly" on a 1/7) is a hold. Ignore the scale legend inside the Verdict line.
8. N/A vs numeric: Tool/Memory (or any dim) whose justification is N/A cannot keep a 1–5. Paste not-applicable and empty turns.
9. Justification turns vs selected turns: if the writeup names Turn N and the selected list is disjoint from those numbers, the list is wrong.
10. Copied A/C justifications and turn-1-only cites on a 7+ turn transcript are holds.

Return a hold list of full task IDs with the check name and one-line reason. Do not skip tasks that look "probably fine".
"""


@dataclass(frozen=True)
class InvariantHit:
    task_id: str
    family: str
    check_ids: tuple[int, ...]
    detail: str


@dataclass
class InvariantReport:
    hits: list[InvariantHit] = field(default_factory=list)

    @property
    def hold_ids(self) -> list[str]:
        return sorted({h.task_id for h in self.hits})

    def summary(self) -> str:
        if not self.hits:
            return "fixer invariants: clean"
        by_family: dict[str, int] = {}
        for hit in self.hits:
            by_family[hit.family] = by_family.get(hit.family, 0) + 1
        counts = ", ".join(f"{k}={v}" for k, v in sorted(by_family.items()))
        return (
            f"fixer invariants: {len(self.hits)} hit(s) on "
            f"{len(self.hold_ids)} task(s) ({counts})"
        )


def form_likert_side(likert: Any) -> str | None:
    """1–3 left, 4 tie, 5–7 right. Spec buckets that call 3/5 ties are wrong."""
    try:
        n = int(likert)
    except (TypeError, ValueError):
        return None
    if n in LEFT_LIKERT:
        return "left"
    if n in TIE_LIKERT:
        return "tie"
    if n in RIGHT_LIKERT:
        return "right"
    return None


def dimension_is_na(dim: DimensionRating) -> bool:
    if dim.not_applicable:
        return True
    just = dim.justification or ""
    if dim.dimension in {"Tool & connector reliability", "Memory & personalization"}:
        return bool(NA_RE.search(just))
    return bool(NA_RE.search(just) and dim.rating is None)


def empty_turns_ok(dim: DimensionRating) -> bool:
    """N/A and a 4 or 5 may have []. Scores 1–3 may not."""
    if dimension_is_na(dim):
        return True
    try:
        return int(dim.rating) in {4, 5}
    except (TypeError, ValueError):
        return False


def mean_dimension_side(
    ratings: Sequence[DimensionRating], left: str, right: str
) -> str | None:
    """Average 1–5 direction. Customer Inconsistent Ranking follows this, not dominance."""
    a: list[int] = []
    b: list[int] = []
    for rating in ratings:
        if dimension_is_na(rating) or rating.rating is None:
            continue
        if rating.model == left:
            a.append(int(rating.rating))
        elif rating.model == right:
            b.append(int(rating.rating))
    if not a or not b:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    if ma > mb + 1e-9:
        return "left"
    if mb > ma + 1e-9:
        return "right"
    return "tie"


def dominance_side(
    ratings: Sequence[DimensionRating], left: str, right: str
) -> str | None:
    by_dim: dict[str, dict[str, int]] = {}
    for rating in ratings:
        if dimension_is_na(rating) or rating.rating is None:
            continue
        if rating.model not in (left, right):
            continue
        by_dim.setdefault(rating.dimension, {})[rating.model] = int(rating.rating)
    a_wins = b_wins = 0
    for scores in by_dim.values():
        if left not in scores or right not in scores:
            continue
        if scores[left] > scores[right]:
            a_wins += 1
        elif scores[right] > scores[left]:
            b_wins += 1
    if a_wins and not b_wins:
        return "left"
    if b_wins and not a_wins:
        return "right"
    return "tie"


def explains_inversion(text: str) -> bool:
    return bool(INVERSION_RE.search(text or ""))


def cited_turns(text: str) -> set[int]:
    return {int(n) for n in TURN_RE.findall(text or "")}


def sxs_body(text: str) -> str:
    """Drop the overlay Verdict line so its scale legend is not a magnitude hit."""
    return "\n".join(
        line for line in (text or "").splitlines() if not line.strip().startswith("Verdict:")
    )


def foreign_slots(text: str, left: str, right: str) -> tuple[str, ...]:
    """Slots named as Model/Response/Product X that are not this comparison."""
    allowed = {left.upper(), right.upper()}
    found = {
        m.group(1).upper() for m in SLOT_MENTION_RE.finditer(sxs_body(text or ""))
    }
    return tuple(sorted(found - allowed))


def retarget_foreign_slots(text: str, left: str, right: str, likert: Any = None) -> str:
    """Rewrite Model B (etc.) to the right-hand slot of this pair.

    Honeybee A-vs-C writeups still say Model B because the form was copied from
    A-vs-B. B is the other product, which is C, not the Likert winner. Replacing
    B with A on a 1–3 produces "Model A is better than Model A".
    """
    if not foreign_slots(text, left, right):
        return text
    target = right

    def repl(match: re.Match[str]) -> str:
        letter = match.group(1)
        if letter.upper() in {left.upper(), right.upper()}:
            return match.group(0)
        replacement = target.upper() if letter.isupper() else target.lower()
        return match.group(0)[: -len(letter)] + replacement

    return SLOT_MENTION_RE.sub(repl, text or "")


def likert_magnitude_ok(likert: Any, text: str) -> bool:
    body = sxs_body(text)
    side = form_likert_side(likert)
    if side is None:
        return True
    try:
        n = int(likert)
    except (TypeError, ValueError):
        return True
    much = bool(MUCH_BETTER_RE.search(body))
    slight = bool(SLIGHTLY_BETTER_RE.search(body))
    if n in (3, 5) and much and not slight:
        return False
    if n in (1, 7) and slight and not much:
        return False
    return True


def is_na_score_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower().replace("_", "-") in NA_SCORE_TOKENS


def _slot_neutral_just(text: str) -> str:
    neutralized = SLOT_LETTER_RE.sub("model X", text or "")
    neutralized = re.sub(r"\b[ABCD]\b", "X", neutralized)
    return re.sub(r"\s+", " ", neutralized).strip().lower()


def copied_justification_pairs(task: Task) -> list[tuple[str, str, str]]:
    """Same-dimension writeups that look copied across slots."""
    by_dim: dict[str, list[DimensionRating]] = {}
    for dim in task.dimension_ratings:
        if dimension_is_na(dim):
            continue
        just = (dim.justification or "").strip()
        if len(just) < COPIED_JUST_MIN_LEN:
            continue
        by_dim.setdefault(dim.dimension, []).append(dim)
    hits: list[tuple[str, str, str]] = []
    for dimension, dims in by_dim.items():
        for i, left in enumerate(dims):
            for right in dims[i + 1 :]:
                ratio = SequenceMatcher(
                    None,
                    _slot_neutral_just(left.justification),
                    _slot_neutral_just(right.justification),
                ).ratio()
                if ratio >= COPIED_JUST_RATIO:
                    hits.append((left.model, right.model, dimension))
    return hits


def ranking_inversion(
    task: Task, comparison: Sxs | None = None
) -> tuple[str, str, str] | None:
    """(mean_side, likert_side, pair_label) when they are opposite winners."""
    pair = comparison or (task.comparison_list()[0] if task.comparison_list() else None)
    if pair is None:
        return None
    likert_side = form_likert_side(pair.likert)
    mean_side = mean_dimension_side(task.dimension_ratings, pair.left, pair.right)
    if likert_side in {"left", "right"} and mean_side in {"left", "right"}:
        if likert_side != mean_side:
            return mean_side, likert_side, pair.pair_label
    return None


def inversion_is_unexplained(task: Task, comparison: Sxs | None = None) -> bool:
    found = ranking_inversion(task, comparison)
    if found is None:
        return False
    pair = comparison or task.comparison_list()[0]
    return not explains_inversion(pair.justification or "")


def assistant_generated_file(sub: ModelSubmission | None) -> tuple[bool, str]:
    if sub is None:
        return False, ""
    if sub.attachments or sub.final_output_uploads:
        return True, "uploads_present"
    if any(
        getattr(entry, "has_model_files", False)
        or getattr(entry, "attachment_names", None)
        for entry in sub.turn_manifest
    ):
        return True, "manifest_has_model_files"
    assistant = "\n".join(t.text for t in sub.conversation if t.role == "assistant")
    if any(t.artifacts_mentioned for t in sub.conversation if t.role == "assistant"):
        return True, "artifacts_mentioned_in_turns"
    for name, pattern in ARTIFACT_PLACEHOLDER_PATTERNS:
        if name in STRONG_PLACEHOLDERS and pattern.search(assistant):
            return True, f"assistant_placeholder:{name}"
    for match in FILE_EXT_RE.finditer(assistant):
        window = assistant[max(0, match.start() - 80) : match.end() + 20]
        if GENERATE_VERB_RE.search(window):
            return True, "assistant_created_named_file"
    return False, ""


def _patch_get(patch: Any, key: str, default: Any = None) -> Any:
    if isinstance(patch, dict):
        return patch.get(key, default)
    return getattr(patch, key, default)


def patched_task(task: Task, patches: Sequence[Any]) -> Task:
    """Copy of `task` with fixer patches applied. Skips failed/empty patches."""
    ratings = [replace(dim) for dim in task.dimension_ratings]
    index = {(d.model, d.dimension): i for i, d in enumerate(ratings)}
    comparisons = [replace(sxs) for sxs in task.comparison_list()]
    by_pair = {sxs.pair_label: i for i, sxs in enumerate(comparisons)}
    submissions = {
        slot: (replace(sub) if sub is not None else None)
        for slot, sub in (
            ("A", task.model_a),
            ("B", task.model_b),
            ("C", task.model_c),
            ("D", task.model_d),
            ("S", task.simplified),
        )
    }

    for patch in patches:
        if _patch_get(patch, "error"):
            continue
        new_value = _patch_get(patch, "new_value")
        if new_value is None:
            continue
        kind = _patch_get(patch, "kind")
        slot = _patch_get(patch, "slot")
        dimension = _patch_get(patch, "dimension")
        comparison = _patch_get(patch, "comparison") or ""
        if kind == "dimension_rating" and (slot, dimension) in index:
            current = ratings[index[(slot, dimension)]]
            if is_na_score_token(new_value):
                ratings[index[(slot, dimension)]] = replace(
                    current, rating=None, not_applicable=True
                )
            else:
                ratings[index[(slot, dimension)]] = replace(
                    current, rating=int(new_value)
                )
        elif kind == "dimension_justification" and (slot, dimension) in index:
            ratings[index[(slot, dimension)]] = replace(
                ratings[index[(slot, dimension)]], justification=str(new_value)
            )
        elif kind == "dimension_turns" and (slot, dimension) in index:
            nums = new_value if isinstance(new_value, list) else [new_value]
            cleaned: list[int] = []
            for item in nums:
                try:
                    cleaned.append(int(item))
                except (TypeError, ValueError):
                    continue
            ratings[index[(slot, dimension)]] = replace(
                ratings[index[(slot, dimension)]], relevant_turns=cleaned
            )
        elif kind in {"preference_likert", "preference_justification"}:
            label = comparison if comparison in by_pair else (comparisons[0].pair_label if comparisons else "")
            if label not in by_pair:
                continue
            current = comparisons[by_pair[label]]
            if kind == "preference_likert":
                comparisons[by_pair[label]] = replace(current, likert=int(new_value))
            else:
                comparisons[by_pair[label]] = replace(
                    current, justification=str(new_value)
                )
        elif kind == "produced_final_outcome" and slot in submissions:
            sub = submissions[slot]
            if sub is not None:
                submissions[slot] = replace(sub, produced_final_outcome=bool(new_value))

    primary = comparisons[0] if comparisons else task.sxs
    return replace(
        task,
        dimension_ratings=ratings,
        comparisons=comparisons,
        sxs=primary,
        model_a=submissions["A"] or task.model_a,
        model_b=submissions["B"] or task.model_b,
        model_c=submissions["C"],
        model_d=submissions["D"],
        simplified=submissions["S"],
    )


def _preference_side_simple(text: str, left: str, right: str) -> str | None:
    lower = sxs_body(text or "").lower()
    if TIE_RE.search(lower):
        return "tie"
    left_win = bool(
        re.search(
            rf"\b(?:model|product|response)\s+{re.escape(left.lower())}\s+"
            rf"(?:is\s+|was\s+)?(?:much |slightly |clearly )?(?:better|ahead|preferred)\b",
            lower,
        )
    )
    right_win = bool(
        re.search(
            rf"\b(?:model|product|response)\s+{re.escape(right.lower())}\s+"
            rf"(?:is\s+|was\s+)?(?:much |slightly |clearly )?(?:better|ahead|preferred)\b",
            lower,
        )
    )
    if left_win and not right_win:
        return "left"
    if right_win and not left_win:
        return "right"
    return None


def _slot_has_assistant_text(sub: ModelSubmission | None) -> bool:
    """Snowflake ingest fills user-only manifest turns. Those are not a transcript."""
    if sub is None:
        return False
    return any(
        t.role == "assistant" and (t.text or "").strip()
        for t in sub.conversation
    )


def audit_task(task: Task) -> list[InvariantHit]:
    hits: list[InvariantHit] = []
    tid = task.task_id

    for sub in task.trajectories():
        generated, reason = assistant_generated_file(sub)
        claimed = sub.produced_final_outcome
        if claimed is False and generated:
            hits.append(
                InvariantHit(
                    tid,
                    "produced_outcome",
                    (82,),
                    f"Model {sub.model} checkbox is No but {reason}",
                )
            )
        # 1–5 fetched exchanges against a 12+ turn chat is a stub. Turn-1-only
        # only fires when max(legal) >= 7, so these used to ship.
        if (
            sub.corroborated_exchanges >= 12
            and sub.exchange_count() <= 5
            and _slot_has_assistant_text(sub)
        ):
            hits.append(
                InvariantHit(
                    tid,
                    "transcript",
                    (310,),
                    f"Model {sub.model}: stub capture {sub.exchange_count()} vs "
                    f"{sub.corroborated_exchanges} corroborated exchanges",
                )
            )
        sibling_ok = any(
            _slot_has_assistant_text(other)
            for other in task.trajectories()
            if other.model != sub.model
        )
        if (
            not _slot_has_assistant_text(sub)
            and sibling_ok
            and sub.corroborated_exchanges >= 12
        ):
            hits.append(
                InvariantHit(
                    tid,
                    "transcript",
                    (310,),
                    f"Model {sub.model}: no assistant transcript; cites cannot be opened",
                )
            )

    for dim in task.dimension_ratings:
        sub = task.submission(dim.model)  # type: ignore[arg-type]
        legal = sub.turn_indices() if sub is not None else set()
        turns = list(dim.relevant_turns or [])
        if dimension_is_na(dim):
            if turns:
                hits.append(
                    InvariantHit(
                        tid,
                        "dimension_turns",
                        (310,),
                        f"{dim.model}/{dim.dimension}: N/A still cites {turns}",
                    )
                )
            if dim.rating is not None:
                hits.append(
                    InvariantHit(
                        tid,
                        "dimension_score",
                        (300,),
                        f"{dim.model}/{dim.dimension}: N/A still scored {dim.rating}",
                    )
                )
            continue
        if not turns:
            if empty_turns_ok(dim):
                continue
            hits.append(
                InvariantHit(
                    tid,
                    "dimension_turns",
                    (310,),
                    f"{dim.model}/{dim.dimension}: empty cites on a scored dimension",
                )
            )
            continue
        selected = set(turns)
        cited = cited_turns(dim.justification)
        if cited and selected and not (cited & selected):
            hits.append(
                InvariantHit(
                    tid,
                    "dimension_turns",
                    (310,),
                    f"{dim.model}/{dim.dimension}: selected {sorted(selected)} disjoint from cited {sorted(cited)}",
                )
            )
        if selected == {1} and legal and max(legal) >= 7:
            hits.append(
                InvariantHit(
                    tid,
                    "dimension_turns",
                    (310,),
                    f"{dim.model}/{dim.dimension}: only cite is turn 1 on a {max(legal)}-turn transcript",
                )
            )
        if legal:
            oor = [n for n in turns if n not in legal]
            if oor:
                hits.append(
                    InvariantHit(
                        tid,
                        "dimension_turns",
                        (310,),
                        f"{dim.model}/{dim.dimension}: out-of-range {oor}",
                    )
                )
            missing_text = [
                n
                for n in turns
                if n in legal
                and _slot_has_assistant_text(sub)
                and not any(
                    t.index == n and t.role == "assistant" and (t.text or "").strip()
                    for t in sub.conversation
                )
            ]
            if missing_text:
                hits.append(
                    InvariantHit(
                        tid,
                        "transcript",
                        (310,),
                        f"{dim.model}/{dim.dimension}: no assistant text at {missing_text}",
                    )
                )

    for left_slot, right_slot, dimension in copied_justification_pairs(task):
        hits.append(
            InvariantHit(
                tid,
                "copied_justification",
                (300,),
                f"{dimension}: Model {left_slot} and Model {right_slot} justifications look copied",
            )
        )

    for sxs in task.comparison_list():
        likert_side = form_likert_side(sxs.likert)
        just_side = _preference_side_simple(sxs.justification, sxs.left, sxs.right)
        if likert_side in {"left", "right"} and just_side == "tie":
            hits.append(
                InvariantHit(
                    tid,
                    "sxs_scale",
                    (400, 470),
                    f"{sxs.pair_label}: Likert {sxs.likert} is a win but writeup reads as a tie",
                )
            )
        if (
            likert_side in {"left", "right"}
            and just_side in {"left", "right"}
            and likert_side != just_side
        ):
            hits.append(
                InvariantHit(
                    tid,
                    "sxs_scale",
                    (400, 470),
                    f"{sxs.pair_label}: Likert {sxs.likert} is {likert_side} but writeup prefers {just_side}",
                )
            )
        if inversion_is_unexplained(task, sxs):
            mean_side, likert_side, label = ranking_inversion(task, sxs)  # type: ignore[misc]
            hits.append(
                InvariantHit(
                    tid,
                    "inconsistent_ranking",
                    (460,),
                    f"{label}: dimension mean favors {mean_side} but Likert {sxs.likert} favors {likert_side}, writeup does not explain the inversion",
                )
            )
        foreign = foreign_slots(sxs.justification, sxs.left, sxs.right)
        if foreign:
            hits.append(
                InvariantHit(
                    tid,
                    "sxs_slot",
                    (400, 470),
                    f"{sxs.pair_label}: writeup names Model {', '.join(foreign)} which is not in this pair",
                )
            )
        if not likert_magnitude_ok(sxs.likert, sxs.justification):
            hits.append(
                InvariantHit(
                    tid,
                    "sxs_magnitude",
                    (400, 470),
                    f"{sxs.pair_label}: Likert {sxs.likert} does not match the better/slightly/much wording",
                )
            )
    return hits


def patches_from_result(result: Any) -> list[Any]:
    if isinstance(result, dict):
        return list(result.get("patches") or [])
    return list(getattr(result, "patches", None) or [])


def audit_fix_results(
    tasks: Sequence[Task],
    results: Sequence[Any],
) -> InvariantReport:
    by_id = {t.task_id: t for t in tasks}
    hits: list[InvariantHit] = []
    for result in results:
        tid = result.get("task_id") if isinstance(result, dict) else result.task_id
        task = by_id.get(str(tid or ""))
        if task is None:
            continue
        status = result.get("status") if isinstance(result, dict) else result.status
        if status not in {"fixable", "skipped"}:
            continue
        patched = patched_task(task, patches_from_result(result))
        hits.extend(audit_task(patched))
    return InvariantReport(hits=hits)


def write_invariant_report(out_dir: Path, report: InvariantReport) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fixer_invariants.json"
    payload = {
        "summary": report.summary(),
        "hold_tasks": report.hold_ids,
        "hits": [
            {
                "task_id": h.task_id,
                "family": h.family,
                "check_ids": list(h.check_ids),
                "detail": h.detail,
            }
            for h in report.hits
        ],
        "subagent_prompt": SUBAGENT_PROMPT.strip(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "fixer_invariants_holds.txt").write_text(
        "\n".join(report.hold_ids) + ("\n" if report.hold_ids else ""),
        encoding="utf-8",
    )
    return path


def load_fixer_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list")
    return data


def merge_fixer_dirs(dirs: Iterable[Path]) -> list[dict]:
    """Later directories win on the same field, matching backfill_steps merge order."""
    by_id: dict[str, dict] = {}
    for directory in dirs:
        results_path = directory / "fixer_results.json"
        if not results_path.exists():
            raise FileNotFoundError(results_path)
        for result in load_fixer_results(results_path):
            tid = str(result.get("task_id") or "")
            if not tid:
                continue
            existing = by_id.get(tid)
            incoming = list(result.get("patches") or [])
            if existing is None:
                by_id[tid] = {
                    "task_id": tid,
                    "status": result.get("status") or "fixable",
                    "patches": incoming,
                }
                continue
            by_key: dict[tuple, dict] = {}
            order: list[tuple] = []
            for patch in list(existing.get("patches") or []) + incoming:
                key = (
                    patch.get("kind"),
                    patch.get("slot"),
                    patch.get("dimension"),
                    patch.get("comparison"),
                    patch.get("criterion_id"),
                )
                if key not in by_key:
                    order.append(key)
                by_key[key] = patch
            existing["patches"] = [by_key[k] for k in order]
            if result.get("status") == "fixable":
                existing["status"] = "fixable"
    return list(by_id.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-csv", type=Path, required=True)
    parser.add_argument(
        "--fixer-dir",
        type=Path,
        action="append",
        required=True,
        help="Fixer output dir. Pass twice to merge overlay on top of the original.",
    )
    parser.add_argument("--project", default="", help="Profile key, e.g. l1_sxs_c")
    parser.add_argument("--from-snowflake", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Hydrate share links from this snapshot cache before auditing. "
             "Without it, ingest only has user-only manifest turns and Transcript "
             "cannot be judged.",
    )
    parser.add_argument("--hydrate-workers", type=int, default=8)
    parser.add_argument(
        "--print-subagent-prompt",
        action="store_true",
        help="Print the checklist a subagent must run on every fixer sheet",
    )
    args = parser.parse_args(argv)
    if args.print_subagent_prompt:
        print(SUBAGENT_PROMPT.strip())
        return 0

    forced: ProjectProfile | None = profile_by_key(args.project) if args.project else None
    if args.from_snowflake:
        tasks, _ingest = load_taskattempts_csv(args.tasks_csv, profile=forced)
    else:
        try:
            tasks, _ingest = load_taskattempts_csv(args.tasks_csv, profile=forced)
        except Exception:
            tasks = load_tasks(args.tasks_csv)
    if forced is None and tasks:
        forced = profile_for_project(tasks[0].project_id)
    del forced

    if args.snapshot_dir:
        from .config import DEFAULT_POLICY
        from .hydrate import hydrate_conversations

        policy = replace(DEFAULT_POLICY, snapshot_dir=str(args.snapshot_dir))
        hyd = hydrate_conversations(
            tasks, policy, workers=args.hydrate_workers
        )
        print(
            f"hydrated {hyd.hydrated}/{hyd.submissions} submissions, "
            f"{hyd.turns} turns",
            file=sys.stderr,
        )
        if hyd.short_captures:
            print(f"short captures: {len(hyd.short_captures)}", file=sys.stderr)

    results = merge_fixer_dirs(args.fixer_dir)
    report = audit_fix_results(tasks, results)
    out_dir = args.out_dir or args.fixer_dir[-1]
    path = write_invariant_report(out_dir, report)
    print(report.summary(), file=sys.stderr)
    print(f"wrote {path}", file=sys.stderr)
    if report.hits:
        for hit in report.hits[:40]:
            print(f"  {hit.task_id}  {hit.family}: {hit.detail}", file=sys.stderr)
        if len(report.hits) > 40:
            print(f"  … {len(report.hits) - 40} more", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
