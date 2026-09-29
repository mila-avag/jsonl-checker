# Eval: system prompt ↔ pull location

For engineering. Profile: **`--project l1_sxs`** (Honeybee L1 SxS queue, project `6a567b80aaa4d140131eae6c`). Step IDs live in `honeybee_qc/projects.py` → `HONEYBEE_L1_SXS`.

There are **four auditor system prompts** in the eval. Every LLM call uses exactly one of them. Deterministic checks have **no** system prompt.

User prompts are built per check (the long instruction + evidence dump). System prompts are stage-wide.

JSON paths are under one Snowflake/Redash row:

```
taskattempts.response.BEFORE[<step_id>].output.<field>
```

In code that is `honeybee_qc/ingest.py` reading `row["response"]["before"]`.

Disabled on this profile (never called): **60, 71, 72, 76**. Unique Links **91** is on.

---

## Folder map

```
honeybee_qc/
│
├── projects.py                          SOURCE OF TRUTH for step IDs
│   └── HONEYBEE_L1_SXS
│
├── ingest.py                            PULL: Snowflake response → Task
├── hydrate.py                           PULL: share URL / HTML export → transcript
├── sources.py                           PULL: browser fetch of final_link
├── transcripts.py                       PULL: parse HTML/MHTML/PDF chat export
├── context.py                           RENDER: Task → numbered conversation text
│                                        (this is what every LLM user prompt embeds)
│
├── 00_no_llm/                           no system prompt
│   ├── preflight.py                     1000 Unauditable
│   ├── links.py                         90 Valid Links
│   └── gates.py                         91 Unique Links
│                                        96 Minimum Turns
│                                        100 Key Turn Identification (turn-1 leg)
│                                        460 Inconsistent Ranking
│
├── 01_rubric_stage/                     SYSTEM #1
│   ├── prompts.py                       SYSTEM_PROMPT
│   └── stages.py                        one call / criterion + one coverage call
│       checks: 200, 210, 230, 240, 250, 260
│
├── 02_rating_stage/                     SYSTEM #2 and #3
│   ├── rating_prompts.py                RATING_SYSTEM_PROMPT          (blind)
│   │                                    ADJUDICATION_SYSTEM_PROMPT    (tie-break)
│   └── rating_stages.py                 270, 300, 400  then optional adj
│
├── 03_informed_stage/                   SYSTEM #4
│   ├── informed_prompts.py              INFORMED_SYSTEM_PROMPT
│   ├── v5_prompt.py                     user-prompt builders only (no 5th system)
│   └── informed_stages.py               70, 75, 80, 85, 95, 110, 220,
│                                        280, 310, 450, 470  (+ 82/86 if collected)
│
└── NOT EVAL
    ├── fixer.py                         SYSTEM_PROMPT  (paste rewriter, after eval)
    └── fill_dims.py                     FILL_SYSTEM_PROMPT
```

---

## The four system prompts

| # | Constant | File | Used by |
|---|---|---|---|
| 1 | `SYSTEM_PROMPT` | `honeybee_qc/prompts.py` | rubric stage (`stages.py`) |
| 2 | `RATING_SYSTEM_PROMPT` | `honeybee_qc/rating_prompts.py` | blind 270 / 300 / 400 |
| 3 | `ADJUDICATION_SYSTEM_PROMPT` | `honeybee_qc/rating_prompts.py` | second opinion on 300/400 disagreements |
| 4 | `INFORMED_SYSTEM_PROMPT` | `honeybee_qc/informed_prompts.py` | every informed check, including v5 user builders |

Blind vs informed is a **data** rule, not a different model:

- **Blind (2, 3):** conversation + the thing being rated. Contributor scores/justifications are stripped. `assert_blind` canaries their text.
- **Informed (4):** conversation **plus** the contributor’s own writing. Judges whether that writing holds up.

---

## Shared pull locations (L1 SxS)

These fields are ingested once, then rendered into many user prompts.

```
response.before/
│
├── step-TextCollection-e8ca527371fe/          prompt
│     output.final_hardened_prompt
│     output.prompt_CUJ
│     output.estimated_turns
│     + input artifact uploads on this step
│
├── dataSourceResults/
│     before:0:data-source-string-prefiller-07f6e8dd43f8     assigned domain (70)
│     before:0:data-source-form-prefiller-41acb4777bed       pre-seeded prompt (75)
│       .final_hardened_prompt
│
├── step-1784053126103-0h5ldq/                 target outcome
│     output.criteria[].title
│
├── step-TextCollection-7be9bf826e8c/          golden answer (L1 SxS only)
│     output.golden_answer | golden_ground_truth_answer | …
│
├── step-RubricCriteriaBuilder-rubric01/       weighted rubric
│     output.criteria[].{id, title, weight, annotations.criterion_category}
│
├── A  (suffix _a)
│     step-TextCollection-29cf069962f8/        run
│       output.*trajectory*                    final share URL (90)
│       output.every_*_a                       per-turn share URLs (91)
│       output.model_a_chat_download[].s3Url   chat export
│       output.final_outputs_a[]               named uploads (95)
│     step-RubricCriteriaBuilder-da9bb8c86eab/ turn manifest
│       output.criteria[].title                user-turn text
│       output.criteria[].annotations
│         turn_number_a, turn_link_a
│         model_generated_files_turn_a
│         turn_attachment_a[]                  more uploads (95)
│         key_turn_index_a, key_turn_justification_a
│     step-RubricCriteriaRating-5665ecba649a/  rubric pass/fail
│       output.responseRatings["response-a"][criterion_id].score     0|1
│     step-RubricCriteriaRating-e2c61a59aef7/  rubric turn cites
│       output.responseRatings["response-a"][criterion_id].title     "Turn N"
│     step-ResponseTextCollection-cb06a191fca3/
│     step-RubricEvaluationViewer-e8bc55c84da9/
│       output.annotations.{prefix}_a
│       output.annotations.{prefix}_justif_a
│       output.annotations.{prefix}_turns_a
│
├── B  (suffix _b)  same shape
│     step-TextCollection-f7304e8fa411/        run  (Spark: often MHTML, no share URL)
│       output.model_b_chat_download
│       output.final_outputs_b
│     step-RubricCriteriaBuilder-0e4891d4e48f/ turn manifest
│     step-RubricCriteriaRating-49853f6eba5a/  scores  response-b
│     step-RubricCriteriaRating-da2090d5dc04/  turn cites
│     step-ResponseTextCollection-2a199aa081bb/
│     step-RubricEvaluationViewer-48999ca811b6/
│
└── step-QuantitativeModelResponseSelector-15407fcb5060/   A-vs-B ranking
      output.preferenceLikert                1–7  (only 4 is a tie)
      output.responseIdx                     0=A  1=B  -1=tie
      output.fieldResponses.preference_justification
```

Dimension `prefix` values (`ingest.py` `DIMENSION_FIELDS`):

```
outcome_quality
communication_quality
trust_grounding
interaction_efficiency
tool_connector_reliability
memory_personalization
collaboration_quality          ingested, not in check 300’s list
```

Transcripts are **not** in Snowflake. After ingest:

1. `hydrate.py` fetches `final_link` (Chrome) and/or parses `model_*_chat_download`.
2. `context.render_conversation` numbers turns into the user prompt.

---

## 00_no_llm/ — no system prompt

| Check | Code | Pull |
|---|---|---|
| **1000** Unauditable | `preflight.py` / `cli.py` | malformed `response`, unknown `criterion_id` in ratings, provenance contradiction |
| **90** Valid Links | `links.py` `evaluate_check_90` | A/B `final_link` from run-step `*trajectory*` fields. Spark B `final_link_B:missing` is expected |
| **91** Unique Links | `gates.py` `evaluate_check_91` | per-turn URLs: run-step `every_*_{a\|b}` **or** manifest `turn_link_{a\|b}`. HTML-B with no share URL is N/A, not fail |
| **96** Minimum Turns | `gates.py` `evaluate_check_96` | hydrated `conversation` length per slot |
| **100** Key Turn Identification | `gates.py` `evaluate_check_100` | key turn index from A (or per-slot on SxS) vs turn 1. Blind auditor pick is a separate non-fail leg |
| **460** Inconsistent Ranking | `gates.py` `evaluate_check_460` | mean of numeric 1–5 dims vs `preferenceLikert` + writeup |

---

## 01_rubric_stage/

**System:** `prompts.py` `SYSTEM_PROMPT`  
**Runner:** `stages.py`  
**User builders:** `prompts.py` `build_criterion_prompt` (per criterion) · `build_coverage_prompt` (once)

| Check | User prompt | Pull into that prompt |
|---|---|---|
| **200** L1 labels | `build_criterion_prompt` | rubric `title` only. Contributor L1/weight **withheld**. Also `final_hardened_prompt`, target-outcome titles (labeled “contributor’s list, not authoritative”) |
| **210** L2 labels | same call | same |
| **230 / 240 / 250** rubric quality 10/15/20% | same call | criterion text + sibling texts for overlap |
| **260** weights | same call | auditor assigns a weight; compared later to contributor `weight` in `gates.py` |
| coverage / missing criteria | `build_coverage_prompt` | full criterion-text list + prompt + target outcome |

Contributor L1/L2/weight are pulled at ingest for **scoring**, not shown to the auditor.

---

## 02_rating_stage/

**System:** `rating_prompts.py` `RATING_SYSTEM_PROMPT`  
**Runner:** `rating_stages.py` `build_rating_requests`  
Contributor ratings must **not** appear in these user prompts (`assert_blind`).

| Check | User prompt | Pull into that prompt | Contributor value (hidden; used only in `gates.py`) |
|---|---|---|---|
| **270** Rubric Evaluation | `build_criterion_rating_prompt` | one criterion **text** + one model’s rendered conversation + golden answer | `responseRatings[response-{a\|b}][id].score` |
| **300** Dimension Correctness | `build_dimension_rating_prompt` | dimension name + rendered conversation | `{prefix}_{a\|b}` integer 1–5 |
| **400** Ranking Disagreement | `build_likert_prompt` | **both** conversations side by side, no Likert | `preferenceLikert` / `responseIdx` |

### Adjudication (only if 300 or 400 disagree)

**System:** `rating_prompts.py` `ADJUDICATION_SYSTEM_PROMPT`  
**User:** `build_dimension_adjudication_prompt` / `build_ranking_adjudication_prompt`  
**Pull:** conversation (wide budget) + the two numeric ratings + contributor justification. Auditor is told **not** to produce a new score.

---

## 03_informed_stage/

**System:** `informed_prompts.py` `INFORMED_SYSTEM_PROMPT` for **every** row below.  
**Runner:** `informed_stages.py` `build_informed_requests`  
v5 builders live in `v5_prompt.py` but they do **not** get a fifth system prompt.

| Check | User-prompt function | File | Pull into that prompt |
|---|---|---|---|
| **70** Domain Relevance | `build_domain_relevance_prompt` | `informed_prompts.py` | `final_hardened_prompt` + assigned domain prefiller |
| **75** Pre-seeded Consistency | `build_prompt_consistency_prompt` | `informed_prompts.py` | seeded prompt vs prefiller `final_hardened_prompt` |
| **80** Target Outcome | `build_target_outcome_prompt` | `informed_prompts.py` | target-outcome titles + conversations |
| **85** Environment Context | `build_environment_context_prompt` | `informed_prompts.py` | prompt + universe/input manifest (`env_context.py`) |
| **95** Output Artifacts | `build_artifact_prompt` then `build_artifact_confirmation_prompt` | `informed_prompts.py` | named files claimed in transcript vs `final_outputs_*` ∪ `turn_attachment_*` names. Count-only shortfall → `not_evaluated` |
| **110** Key Turn Justification | `build_key_turn_justification_prompt` | `informed_prompts.py` | `key_turn_justification_{a\|b}` + that turn’s transcript text |
| **220** Rubric Autofail | `build_autofail_prompt` (+ second opinion) | `informed_prompts.py` | full rubric texts |
| **280** Rubric relevant turns | `build_criterion_turns_prompt` | `informed_prompts.py` | score-0 (and some score-1) criteria + cited `Turn N` + conversation |
| **310** Dimension relevant turns | `build_dimension_turns_prompt` | `informed_prompts.py` | `{prefix}_justif_*` + `{prefix}_turns_*` + conversation |
| **450** Justifications | `build_dimension_justification_prompt` | `informed_prompts.py` | all seven `{prefix}_justif_{a\|b}` + conversations |
| **470** Verdict | `build_ranking_justification_prompt` + `build_verdict_prompt` | `informed_prompts.py` | `preference_justification` + Likert + both conversations |

### V5 user builders (same system #4). Off on L1 SxS except 82/86 if the field exists.

| Check | User-prompt function | File | Pull | L1 SxS |
|---|---|---|---|---|
| **60** Prompt clarity | `build_prompt_clarity_prompt` | `v5_prompt.py` | `final_hardened_prompt` + golden answer | **disabled** |
| **71** Reasoning requirement | `build_reasoning_requirement_prompt` | `v5_prompt.py` | same | **disabled** |
| **72** Constraints | `build_constraints_prompt` | `v5_prompt.py` | same | **disabled** |
| **76** Prompt request | `build_prompt_request_prompt` | `v5_prompt.py` | same | **disabled** |
| **82** Produced outcome | `build_final_outcome_claim_prompt` | `v5_prompt.py` | `model_produced_final_outcome_{a\|b}` on run step vs files | field usually absent on L1 |
| **86** Known information | `build_known_info_prompt` | `v5_prompt.py` | `step-TextCollection-0c0626b307df` | Aspirational only |

---

## Check → system prompt (cheat sheet)

```
no LLM ........ 90 91 96 100 460 1000
#1 rubric ..... 200 210 230 240 250 260
#2 blind ...... 270 300 400
#3 adjudicate . (optional follow-up on 300/400)
#4 informed ... 70 75 80 85 95 110 220 280 310 450 470
                60 71 72 76 82 86   (v5; 60/71/72/76 off on l1_sxs)
```

---

## Wiring (who calls whom)

```
cli.py
  load_tasks / ingest.py          pull Snowflake
  hydrate.py                      pull chats
  evaluate_check_90/91/96/100/460 no LLM
  run_rubric_stage   → stages.py           system #1
  run_rating_stage   → rating_stages.py    system #2 then #3
  run_informed_stage → informed_stages.py  system #4
  gates.py                        compare auditor JSON to pulled contributor values
```

Cache key (`llm.py` / `cache.py`) includes **system + user + schema + model**. Changing a system prompt busts the cache for that stage.

---

## Other `--project` values

Same four system prompts. Only step IDs change (`projects.py`):

| key | extra pulls |
|---|---|
| `l1` | older A manifest `bda5cf5e772d`; ranking `cd151dbf1b55`; 91 disabled |
| `l1_sxs` | **this eval** |
| `aspirational` | Model C + simplified S; known-info step; produced-outcome checkboxes; V5 60/71/72/76 **on**; 70/75/96/220/460 **off** |
| `aspirational_d` | Model D on the old simplified step |
