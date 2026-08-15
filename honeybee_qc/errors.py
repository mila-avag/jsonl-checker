"""Error-code table, reproduced verbatim from tab 2.

Three entries have no row in tab 2. 85's and 75's strings are the ones human QC
actually writes -- 85's from the audit workflow tab's requirement, 75's from the
customer's own field usage under a label tab 2 never assigned a position. 96's is
provisional: nothing in the QC export uses it, so it is ours until the customer
confirms it.

These strings are the join key against the customer's own tooling. Two of them
are malformed -- 240's non-fail label carries a stray "+" -- and are reproduced
as-is on purpose. Normalising them would silently break downstream matching.

280 and 310 deliberately share both of their strings; they are separate
dimensions with separate scores, so findings are namespaced by check ID
everywhere else and only the emitted label collides.
"""

from __future__ import annotations

ERROR_CODES: dict[int, dict[str, str]] = {
    # 70 is net-new, so its strings come from the auditors' own usage in the QC
    # export rather than from a tab-2 cell. The asymmetry is theirs: the fail
    # string carries a trailing "Prompt", the non-fail string does not.
    70: {
        "fail": "[Fail - Domain Relevance Prompt]",
        "non_fail": "[Non-Fail - Domain Relevance]",
    },
    # Net-new, like 70, and cited by human QC under its own bracketed label with
    # no tab-2 row. The two bands are inconsistently pluralised in the QC export
    # itself -- the fail label reads "Prompts" and the non-fail label
    # "Prompt" -- and both are reproduced exactly as cited, the same discipline
    # 240's stray "+" gets: normalising either would break the join against the
    # customer's own tooling.
    75: {
        "fail": "[Fail - Pre-seeded/Opening Prompts Inconsistency]",
        "non_fail": "[Non-Fail - Pre-seeded/Opening Prompt Inconsistency]",
    },
    80: {"non_fail": "[Non-Fail - Misaligned Target Outcome]"},
    # Not from the Dimension Definition tab. The audit workflow tab requires the
    # prompt to reference "only the entities and events in the task's universe"
    # without giving it a row or a code, and human QC cites it under this string.
    85: {"fail": "[Fail - Environment Context Violation]"},
    90: {"fail": "[Fail - Missing/Invalid Links]"},
    95: {"fail": "[Fail - Artifacts Not Preserved]"},
    # PROVISIONAL. Also not from the Dimension Definition tab -- the audit
    # workflow tab requires "at least 7 turns" without giving it a row or a code
    # -- but unlike 85 this string has no precedent: none of the 322 distinct
    # labels in the human QC export names a turn count, so this one is our
    # invention. The customer must confirm it before anything joins on it. The
    # nearest existing label is "[Fail - Insufficient Trajectory]", which the
    # export uses for something else; adopting it on a guess would corrupt that.
    96: {"fail": "[Fail - Insufficient Turns]"},
    100: {
        "fail": "[Fail - First Turn is Key Turn]",
        "non_fail": "[Non-Fail - Misidentified Key Turn]",
    },
    110: {"non_fail": "[Non-Fail - Incorrect Key Turn Justification]"},
    200: {
        "fail": "[Fail - L1 Labels/Annotations]",
        "non_fail": "[Non-Fail - L1 Labels/Annotations]",
    },
    210: {"non_fail": "[Non-Fail - L2 Labels/Annotations]"},
    220: {"fail": "[Fail - Rubric Autofail]"},
    230: {
        "fail": "[Fail - 10%+ Major Rubric Errors]",
        "non_fail": "[Non-Fail - < 10% Major Rubric Errors]",
    },
    240: {
        "fail": "[Fail - 15%+ Major/Moderate Rubric Errors]",
        "non_fail": "[Non-Fail - < 15%+ Major/Moderate Rubric Errors]",
    },
    250: {
        "fail": "[Fail - 20%+ Major/Moderate/Minor Rubric Errors]",
        "non_fail": "[Non-Fail - < 20% Major/Moderate/Minor Rubric Errors]",
    },
    260: {
        "fail": "[Fail - Criteria Weights]",
        "non_fail": "[Non-Fail - Criteria Weights]",
    },
    270: {
        "fail": "[Fail - Egregious Rating Disagreement]",
        "non_fail": "[Non-Fail - Minor Rating Disagreement]",
    },
    280: {
        "fail": "[Fail - Incorrect Relevant Turns]",
        "non_fail": "[Non-Fail - Incorrect Relevant Turns]",
    },
    300: {
        "fail": "[Fail - Dimension Rating Disagreement]",
        "non_fail": "[Non-Fail - Dimension Rating Disagreement]",
    },
    310: {
        "fail": "[Fail - Incorrect Relevant Turns]",
        "non_fail": "[Non-Fail - Incorrect Relevant Turns]",
    },
    400: {
        "fail": "[Fail - Major Ranking Disagreement]",
        "non_fail": "[Non-Fail - Loose Ranking Disagreement]",
    },
    450: {
        "fail": "[Fail - Bad Justifications]",
        "non_fail": "[Non-Fail - Bad Justifications]",
    },
    460: {"fail": "[Fail - Inconsistent Ranking]"},
    470: {"fail": "[Fail - Lacks Verdict]"},
    # Not a bracketed tag in the spreadsheet. Kept verbatim.
    1000: {"fail": "This task is unauditable"},
}

CLEAN_STATE_TEXT: dict[int, str] = {
    230: "No major issues",
    240: "No major or moderate issues",
    250: "No major, moderate, or minor issues",
}


def error_code(check_id: int, band: str) -> str | None:
    """Return the verbatim code for a (check, band), or None for clean bands."""
    if band in ("clean", "not_evaluated"):
        return None
    try:
        return ERROR_CODES[check_id][band]
    except KeyError as exc:
        raise ValueError(
            f"check {check_id} has no error code for band {band!r}"
        ) from exc


def is_known_code(code: str) -> bool:
    return any(code in bands.values() for bands in ERROR_CODES.values())


def all_codes() -> list[str]:
    return [c for bands in ERROR_CODES.values() for c in bands.values()]
