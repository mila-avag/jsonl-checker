---
name: system-prompt-writing
description: >-
  Writes and rewrites LLM system prompts for rubric generators, coverage
  auditors, judges, and other eval workers. Use when the user pastes a system
  prompt, asks to make a prompt better, add examples, tighten rules, show a
  diff, write a rubric-criteria generator, write a coverage auditor, or share
  prompt-writing rules with the team.
---

# System prompt writing

Rewrite the prompt. Do not lecture about prompting in the abstract.

Default deliverable, in this order:

1. The full replacement prompt, copy-paste ready.
2. A unified diff (or a compact changed/added/removed map if the rewrite is total).
3. The three changes that will move quality, each in one sentence.

Keep the caller's output schema if a parser or eval harness already reads it (same section headers, verdict names, field names, emoji if they are load-bearing). Improve the judging rules; do not silently rename `NEEDS WORK` to `PARTIAL`.

## Diagnose before rewriting

Read the pasted prompt and name the failure mode. Typical ones:

| Smell | What the model will do |
|---|---|
| Same rule said 4+ times | Ignore it and still wrap JSON in fences |
| "implicit" / "complete" / "high quality" with no kill list | Invent asks so two auditors extract different denominators |
| "clearly and verifiably" with no BAD example | Write "The response is accurate" and call it coverage |
| Overlapping verdict bands | FAIL and NEEDS WORK fire on the same defect; labels become a coin flip |
| Negatives defined as "errors" | Emit the inverse of every positive |
| Weights with no bands | A bullet-format item outranks the factual answer |
| No worked example | The model cannot see the shape of a legal item |

If the prompt is for **criteria / rubrics**, also read [examples.md](examples.md). For the audit checklist, read [checklist.md](checklist.md). The team's rubric coverage judge (single prompt + GTFA + criteria; no target outcomes, no multi-turn) is [rubric-eval.md](rubric-eval.md).

## Hard rules (put these in the rewritten prompt)

Say each once, then show a KEEP / KILL pair. Do not nag.

1. **Output contract once.** Schema, types, banned extra text. Then work. Repeat the "JSON only" line at most once at the end.
2. **Self-contained.** A later grader sees only the prompt, readable files, the gold, and the response. Titles name the value, entity, file, or behavior. Ban "correct", "accurate", "appropriate" unless the title also states what that is.
3. **Atomic, with a brake.** One focus per item. Unrelated ideas split. Several details under one focus stay (two named section titles; one comparison that explains a single fact).
4. **Positive framing.** "The response …". Ban does-not / fails-to / avoids / omits. Missing X is failing the matching positive, not a negative item.
5. **Negatives are a different error.** 2–3 typical, never more than 5 unless the task is error-hunting. Illegal: inverse of a positive. Legal: wrong table, bid-vs-ask, wrong speaker, forbidden action, unit mixup.
6. **Necessary implicit only.** An implicit ask exists only if the explicit ask is wrong or incomplete without it (units, which entity/table/year, required side effect). Tone, polish, extra charts, "show work", and unrequired citations are not asks.
7. **Gold is a source of values, not a second prompt.** Restatements of the user ask are not new items. Concrete numbers, names, decisions, quotes, and ABSENT items are. If gold contradicts the user prompt, the prompt wins and the contradiction is a defect.
8. **Negatives never cover a produce-ask.** A penalty for doing the wrong thing is not coverage for producing the right thing. Negatives cover ABSENT / commission-error asks only.
9. **Verdicts are a first-match tree.** FAIL, NEEDS WORK / PARTIAL, and PASS must not share a defect. Write "first match wins" and list FAIL causes first.
10. **Weights have bands, not vibes.** Defining ask 8–10, key support 5–7, useful 3–4, nice-to-have 1–2. Primary ask at ≤2 is underweighted. Cosmetic item at ≥7 is overweighted unless the prompt itself made that format load-bearing — and even then it may not outrank the factual answer.
11. **Denominators are reproducible.** Define Y. If a class is empty, `0 / 0`. Implicit Y = necessary implicits only. Do not let the model grow Y to look thorough.
12. **Paraphrase is coverage.** Do not fail a title because the auditor would have worded it differently, if a grader could still apply it.

## Rewrite workflow

1. Extract the job (generator vs auditor vs judge), the inputs, the output schema, and any harness constraints.
2. Write the new prompt in this order: role → inputs → definitions with kill lists → hard rules → KEEP/KILL examples → weight bands → verdict tree → output schema → one closing contract line.
3. Include at least one fully worked mini-example in the *same domain as the task*, not a generic "includes X / wrong table" stub.
4. Show the diff. Call out the three load-bearing changes.
5. Do not convert their scale (1–10 vs 1–5, PASS/FAIL names) unless they asked.

## What not to do

- Do not add persona fluff, "take a deep breath", or five restatements of the output format.
- Do not invent extra eval dimensions the caller did not ask for.
- Do not turn a coverage auditor into a naturalness / prompt-quality judge.
- Do not replace their parser-facing Markdown with a new schema.
- Do not write the skill's own rules into a prompt that is not about rubric/coverage work. Use the *method* (contract, examples, exclusive bands), not the Northwind story, for unrelated jobs.

## Sharing this skill

It lives at `.cursor/skills/system-prompt-writing/`. Anyone who pulls the repo gets it. Point teammates at this folder; do not paste the full prompt-writing rules into Slack.
