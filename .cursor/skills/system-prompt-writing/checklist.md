# System prompt audit checklist

Run this against a draft before shipping. Fail the draft if any **blocker** is open.

## Blockers

- [ ] Output contract is stated once (schema + banned extra text), not four times.
- [ ] Every scored item is checkable without web search, code execution, listening to audio, or re-deriving a gold value.
- [ ] Concrete values/names from the gold appear in titles. "The correct X" is illegal.
- [ ] Implicit / complete / thorough has a kill list, not just a definition.
- [ ] Verdict bands are exclusive. First-match tree. FAIL listed first.
- [ ] Negative items are commission errors, not inverses, not "missing X".
- [ ] If the output is `X / Y`, Y is defined and empty classes are `0 / 0`.

## Coverage / ask extraction (auditors)

- [ ] EXPLICIT = stated in the user prompt, or the question cannot be answered without it.
- [ ] IMPLICIT = only what makes an explicit ask wrong if missing.
- [ ] [GTFA] = gold-only facts, decisions, quotes, ABSENT items. Restatements of the prompt are not new items.
- [ ] Prompt wins if gold contradicts it; the contradiction is recorded.
- [ ] A title covers an ask only if *passing it would satisfy that ask*.
- [ ] Negatives do not cover produce-asks.
- [ ] Orphans (title matches no ask) and contradictions (title rewards the forbidden/wrong thing) are defined.

## Quality of titles (generators and auditors)

- [ ] Positive framing: "The response …".
- [ ] Atomic: one focus. Keep-as-one cases exist so models do not over-split.
- [ ] Final state, not process, unless the prompt required the method.
- [ ] Named subject when several entities/tables/years exist.
- [ ] Ban-list for vague words: professional, clear, concise, high-quality, well-structured, comprehensive, thorough, helpful, insightful, accurate/correct (unless the value is in the title).
- [ ] Full overlap = duplicate. A subset is not a duplicate.

## Weights

Use the caller's scale. If they use −10..10:

| | Positive | Negative |
|---|---|---|
| Defining / catastrophic | 8–10 | −8 to −10 |
| Key support / serious wrong-source | 5–7 | −5 to −7 |
| Useful constraint | 3–4 | −3 to −4 |
| Nice-to-have / minor contamination | 1–2 | −1 to −2 |

- [ ] Never 0, never a float.
- [ ] Primary ask at |w| ≤ 2 → UNDERWEIGHTED.
- [ ] Cosmetic extra at |w| ≥ 7 → OVERWEIGHTED, unless the prompt made that format load-bearing.
- [ ] A load-bearing format constraint may outrank a *supporting* fact. It may not outrank the factual answer.
- [ ] Total |negative| > total positive is inverted emphasis unless the prompt is about error detection, avoidance, or safety.
- [ ] 2–3 negatives typical; >5 is a smell.

## Verdict tree (paste and fill)

First match wins:

1. **FAIL** if: any EXPLICIT ask uncovered; a title contradicts prompt or gold; fewer than half of extracted asks covered; the core ask is missing from the set.
2. **NEEDS WORK** if not FAIL, and: a necessary implicit or GTFA item is uncovered; quality flags; a bad negative; weight misalignment.
3. **PASS** only if: every EXPLICIT, necessary IMPLICIT, and GTFA required item is covered by a checkable title; no contradictions; negatives are specific and not inverses; no critical quality flag on a title that is the *sole* coverage of an ask.

Reasonable paraphrase is coverage.

## Unreadable inputs

If the prompt refers to audio, video, or a file the model cannot read: still write criteria for what the prompt asks about that file. Put unknowns in `[square brackets]`. Do not pretend the file was heard. Do not drop the items.
