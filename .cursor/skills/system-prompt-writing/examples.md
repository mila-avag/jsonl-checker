# Worked examples

Use these inside rewritten prompts. Change the domain to match the task; keep the shape.

## Generator — KEEP / KILL

Prompt: "Using the Q3 table, what was Northwind's operating margin, and should we expand the Midwest region?"
Gold: "Northwind's Q3 operating margin was 18.4%. Do not expand the Midwest; Midwest revenue fell 12% year over year."

KEEP:

```text
The response states Northwind's Q3 operating margin as 18.4%.                          +9
The response recommends against expanding the Midwest region.                         +8
The response cites a 12% year-over-year decline in Midwest revenue as the reason.     +6
The response reports East region revenue as if it were Midwest revenue.               -6
The response treats the 12% revenue decline as a 12 percentage-point margin change.   -5
```

KILL → rewrite:

| BAD | WHY | GOOD |
|---|---|---|
| The response gives the correct operating margin. | Not self-contained | …states Northwind's Q3 operating margin as 18.4%. |
| The response includes the margin and a recommendation and a YoY comparison. | Three tests | Split |
| The response does not recommend expanding the Midwest. | Negative framing of a required behavior | …recommends against expanding the Midwest region. (+8) |
| The response is clear and professional. | Vague | Drop |
| The response calculates the correct tax return. | Grader cannot compute | …states the tax return as $1,000. |
| The response shows the company's cost is $500. | Unspecified subject | …states Acme's shipping cost as $500. |
| "Mentions X and Y" plus "Mentions X and Y in the summary" | Full overlap | Keep one |
| Two titled sections AND APA citations in one title | Unrelated foci | Split |

KEEP as ONE item (single focus, several details):

```text
The response includes two sections, one titled Risks and one titled Next steps.
The response explains that the put is cheap because it is ~40% OTM, while the call is only ~13% OTM.
```

Illegal inverse pair:

```text
+ The response states the closing price as $142.50.
− The response does not state the closing price as $142.50.
```

Legal pair (different error):

```text
+ The response states the closing price as $142.50.
− The response reports the opening price as if it were the closing price.
```

## Auditor — ask extraction

Same Northwind prompt + gold, plus "Reply in two bullets."

Asks (this is the whole list):

```text
[EXPLICIT] State Northwind's Q3 operating margin
[EXPLICIT] Recommend whether to expand the Midwest
[EXPLICIT] Format the reply as two bullets
[IMPLICIT] Use the Q3 table
[GTFA]     Margin value is 18.4%
[GTFA]     Do-not-expand, supported by 12% YoY Midwest revenue decline
[GTFA][ABSENT] East region figures are not used as Midwest
```

Not asks: be concise, sound executive, chart all regions, show work, cite sources.

Coverage:

| Title | Covers? |
|---|---|
| The response states Northwind's Q3 operating margin as 18.4%. | Yes — explicit + GTFA value |
| The response answers the question. | No |
| The response is accurate. | No |
| The response gives the correct margin. | No — value missing |
| The response does not omit the margin. (−) | No — inverse; cannot cover a produce-ask |
| The response uses Python. | No — process, unless the prompt required Python |

## Auditor — verdict

- Missing "state the margin" → **FAIL** (uncovered explicit / missing core).
- Margin covered, "two bullets" missing, one inverse negative → **NEEDS WORK**.
- All necessary asks covered by checkable titles, two legal negatives, core at weight 9 → **PASS**.
- Gold says 18.4% and a title requires 14% → **FAIL** (contradiction), even if every explicit ask is named.

## Mini rubric shape (generator output)

```json
[
  {"id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "title": "The response states Northwind's Q3 operating margin as 18.4%.", "weight": 9, "annotations": {}},
  {"id": "b2c3d4e5-f6a7-8901-bcde-f12345678901", "title": "The response recommends against expanding the Midwest region.", "weight": 8, "annotations": {}},
  {"id": "c3d4e5f6-a7b8-9012-cdef-123456789012", "title": "The response cites a 12% year-over-year decline in Midwest revenue as the reason not to expand.", "weight": 6, "annotations": {}},
  {"id": "d4e5f6a7-b8c9-0123-def0-234567890123", "title": "The response uses figures from the Q3 table.", "weight": 5, "annotations": {}},
  {"id": "e5f6a7b8-c9d0-1234-ef01-345678901234", "title": "The response reports East region revenue as if it were Midwest revenue.", "weight": -6, "annotations": {}},
  {"id": "f6a7b8c9-d0e1-2345-f012-456789012345", "title": "The response treats the 12% revenue decline as a 12 percentage-point change in operating margin.", "weight": -5, "annotations": {}}
]
```
