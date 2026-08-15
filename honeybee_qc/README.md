# HoneyBee_v2 QC audit engine

Audits a contributor's side-by-side evaluation work product against the 21 custom
dimensions of project `6a70ffe56999de9083413f7d`.

**Phase 1** (no model calls): preflight validation, check 90, check 100's turn-1
structural fail, check 460, link-versus-PDF provenance, and **all** counting and
threshold arithmetic for every gate.

**Phase 2** (one model call per criterion, ~21 per task): checks 200, 230, 240,
250, 260.

**Phase 3** (a blind pass, ~53 calls per task): checks 270, 300, 400.

323 tests, none of which contact a model, all running in under a second.

**[`PROMPTS.md`](PROMPTS.md) is every prompt the audit sends, rendered byte-exact
from the code** by `python3 -m honeybee_qc.dump_prompts`. Read that to review the
prompts; it cannot drift from what the CLI actually sends.

```bash
python3 -m pytest honeybee_qc/tests -q
python3 -m honeybee_qc.cli tasks.jsonl --out report.json
python3 -m honeybee_qc.cli tasks.jsonl --verify-provenance --snapshot-dir snaps
python3 -m honeybee_qc.cli tasks.jsonl --rubric-stage --rating-stage --cache-db .cache/qc.db
python3 -m honeybee_qc.cli tasks.jsonl --rubric-stage --rating-stage --dry-run  # count calls first
python3 -m honeybee_qc.demo_rubric_stage          # live run on a planted rubric
python3 -m honeybee_qc.demo_rating_stage          # live blind pass on a real share link
```

## Why this order

The thresholds are the actual product. They are also the cheapest thing to get
right and the only part testable to certainty, so they were built first and
tested exhaustively at every boundary. A model stage that finds issues perfectly
is worthless if the arithmetic turning those issues into a percentage is off by
one criterion.

## Design invariants

**Models find issues; Python counts them.** No threshold in this codebase is
evaluated by a language model. Gates are pure functions over plain findings, so
every percentage in a customer report is reproducible from the finding list.

**Auditor independence.** Almost every dimension is a disagreement measurement.
Judgments must be formed before the contributor's values enter context; anchoring
does not merely add noise, it biases every rate toward agreement. This is why no
two judgments ever share a model call, and why the blind checks (270, 300, 400)
are kept in a separate pass from the checks that judge the contributor's own
citations and justifications (280, 310, 450).

**An unauditable judgment abstains rather than guessing.** The transcript is not
the work product: when a model delivers a video, the turn's text collapses to
`Your video is ready!` and the deliverable is unreachable. A guess there produces
a disagreement rate made of noise on exactly the criteria that matter most, so
`cannot_determine` leaves the denominator and is reported.

**Shape is enforced in code, not just in prompts.** A Shape B check cannot
express a fail, so it cannot hallucinate one. `build_verdict` raises
`ShapeViolation` rather than silently accepting an out-of-shape band.

| Shape | Bands available | Checks |
|---|---|---|
| A (binary) | fail, clean | 90, 95, 220, 460, 470, 1000 |
| B (non-fail only) | non_fail, clean | 80, 110, 210 |
| C (graded) | fail, non_fail, clean | 100, 200, 230, 240, 250, 260, 270, 280, 300, 310, 400, 450 |

Score 2 is never emitted by anything.

**Counting counts criteria, not issues.** A criterion with three major issues
contributes 1 to the 230 numerator, not 3. Because inclusion widens across
230 → 240 → 250, the numerator is non-decreasing; this is asserted, not assumed,
and a regression raises `CensusCorrupted` rather than publishing a number.

**Denominators are stated on the face of every table.** Unlabeled criteria leave
200's denominator, dimensions both sides mark N/A leave 300's, missing criteria
raise numerators without raising denominators, and unauditable tasks are excluded
from every quality rate.

## Policy decisions

The spec has known gaps. Every one is an explicit field in `config.Policy`, is
recorded in each report for reproducibility, and can be changed without touching
a gate. Defaults:

| Gap | Decision | Basis |
|---|---|---|
| Weight bucket mapping (260) | `low={1,2} medium={3} high={4,5}` | The customer weight guide: 1 minor positive, 2 moderately important, 3 important, 4 highly important, 5 essential |
| Dimension rating scale (300) | 1–5 | The authoring form's `<dim>_<a\|b>` fields are `number 1-5`. Was an assumed 1–10, which understated every disagreement |
| Non-fail floor (240) | Any nonzero rate below the fail threshold | 240 reads "between 1% and 15%" while 230/250 read "more than 0%"; a literal 1% floor leaves a single issue in a small rubric unrepresentable |
| 270 denominator | Judgment count (criteria × models) | Disagreements are counted across both models, so the denominator should match |
| Missing turns (280/310) | Counted as incorrect, tracked under their own category | The spec counts incorrectly selected turns; a score-0 criterion citing nothing is still a citation defect |
| L2 taxonomy (210) | Scored against the 26-leaf form taxonomy | The appendix was missing from the spec export, but the authoring form defines the same taxonomy |
| L1 label set (200) | 8 labels, not the spec's 7 | The form lets contributors pick "Tool and Connector Reliability"; auditing against 7 would manufacture disagreements no contributor could avoid |
| Preference encoding (400/460) | Spec's bipolar 1–7, direction read from `responseIdx` | Confirmed against 73 graded tasks. The spec's bucket boundaries are wrong, see below |
| Weight scale (260) | 1–5, out-of-scale values counted as maximally wrong | The spec says 1–5 with no negatives; live rubrics carry −3 to −5. Voiding those tasks would discard every other check |
| Weight comparison (260) | The auditor returns the range of weights the customer's definitions can support; only a weight outside that range is counted | A weight is a judgement call with a defensible band. Comparing the contributor's point to the auditor's flagged 17 of 17 tasks at a median 53% of criteria, against a handful of human QC citations, because the auditor's taste sits a level lower on the scale (it picked 5 on 22 criteria where contributors picked it on 148) |
| Same-bucket out-of-band weight (260) | Non-fail, never a fail | A 4 against a band of 5 is zero levels so it cannot reach a fail, but it is still a weight the definitions do not support |
| N/A mismatch (300) | Counts toward total disagreements, not major | It is a disagreement with no numeric delta |
| 450 thresholds | Counted within a single justification | "2 or more claims lack evidence" reads as a property of one justification |
| not_evaluated and "clean" | Does not deny a task clean | Otherwise no task is ever clean while 210 is blocked |

## Turn indexing

Declared once and asserted in preflight, because check 100 hard-fails on "turn 1"
and an off-by-one manufactures fails:

- indices are 1-based
- a user message and the response answering it **share** one index, so an index
  identifies an exchange — this matches the rendered share pages, which expose
  one user block and one response block per exchange
- the first turn is a user turn
- `key_turn` must resolve to an index that has an assistant turn

Any adapter for a differently numbered source must renumber, not reinterpret.

## Links

Check 90 is fully deterministic. Three providers are recognised:
`gemini.google.com/share/…`, `share.gemini.google/…`, `chatgpt.com/share/…`,
`chat.openai.com/share/…`, `claude.ai/share/…`.

Contributors file a link per turn plus a final link. Only the **final** link
failing is a 90 fail; invalid per-turn links warn. Private conversation URLs
(`/app/`, `/c/`, `/chat/`) are rejected — they will not open for a reviewer.
HTTP reachability is opt-in and a network timeout must never fail a task.

## Provenance: does the link match the PDF?

The audit reads the contributor's PDF, so nothing stops someone pasting a
well-formed share URL next to a transcript that never happened. With
`--verify-provenance` the engine renders each final link in headless Chrome,
archives the DOM, and compares it to the uploaded PDF.

Comparison uses 5-word shingle containment, not a diff ratio. A PDF is a print
of the page: it carries page furniture, wrapped and hyphenated lines, and lost
role markers, so an exact or order-sensitive comparison would fail on honest
submissions. Matching a 5-word window means the same words appeared in the same
order, which coincidental topical overlap does not produce.

Only the **forward** direction convicts — the fraction of the *live* conversation
present in the PDF. A low value means the PDF is not a transcript of that link.
A PDF holding *extra* material is only `inconclusive`, because appendices,
attachments, and both models in one file are all legitimate.

| Verdict | Meaning | Consequence |
|---|---|---|
| `verified` | ≥80% of link turns present in the PDF | none |
| `inconclusive` | between 40% and 80%, or the PDF carries much extra material | review flag |
| `unverifiable` | fetch failed, no PDF, no browser, too few turns recovered | review flag |
| `contradicted` | ≤40% of the live conversation appears in the PDF | **1000, unauditable** |
| `link_dead` | the page reports the conversation as unavailable | **90 fail** |

Three rules make this safe to run unattended, because a false fraud accusation
costs far more than a missed one:

1. **A fetch failure is never evidence.** Timeouts, missing drivers, and auth
   walls all land in `unverifiable`. Only a page that definitively reports itself
   gone can fail check 90.
2. **There is a deliberate abstention band.** Anything between the thresholds
   goes to a human instead of being guessed at.
3. **Contradiction routes to unauditable, not to a quality fail.** If the PDF and
   the link disagree, no finding drawn from either is trustworthy — and it keeps
   fabricated work out of the defect denominators rather than diluting them.

Batch-level reuse is also detected. Every task is a distinct conversation, so the
same share page or the same PDF appearing under two task IDs is copy-paste rather
than coincidence and is treated as unauditable. Reuse across the two model slots
of a single task is a milder finding: the comparison only contains one
conversation.

Verified against the real Gemini share page: 16 exchanges recovered, an honest
print scored 100% forward containment, and a fabricated PDF that kept the genuine
opening exchange still scored 13% and was correctly contradicted.

### Operational notes

Rendered DOMs are archived to `--snapshot-dir` with a sidecar JSON recording the
URL, share ID, fetch time, turn count, and digest, so a fraud finding stays
reproducible after the page changes or expires. Snapshots double as the fetch
cache. `PROVIDER_MARKERS` in `sources.py` is vendor DOM and will drift; each
provider has an inner content marker set preferred for clean text and an outer
container fallback that survives redesigns.

## The rubric stage (phase 2)

One model call per criterion, plus one task-level coverage pass. The call returns
three things at once — the criterion's issues, an L1 label, and a weight — which
covers checks 200, 230/240/250, and 260 without a second pass.

**Independence is structural.** The auditor is asked to assign a label and a
weight, and the contributor's label and weight are simply absent from the prompt.
Sibling criteria appear (so `overlapping` is detectable) but stripped of their
labels and weights. `assert_independent` re-checks each prompt at runtime and
refuses to send one that binds a contributor value to a criterion ID, because a
leak would bias every rate toward agreement while still looking like it works.

**Validation is strict and one-directional.** An issue is dropped unless its
category is in the closed set and its evidence quote actually appears in the
rubric. Dropping under-reports defects, which is the safe direction; passing
through an invented category would put an unexplainable string in a customer's
error-code column. Every drop is counted and reported.

**Failed criteria leave the denominator.** If a call fails after retries, that
criterion was not audited, so it is excluded from the denominator rather than
counted as clean, and the affected verdicts are marked low confidence.

The model runs with tools disabled. The predecessor engine granted Bash, Read,
Glob, and Grep and let the model explore a filesystem; here all evidence is
stuffed into the prompt, so there is nothing to explore and nothing to vary
between runs.

### Caching

Keyed on prompt, schema, system prompt, model, effort, and policy version
together, via one function that both reads and writes call. The predecessor
engine wrote `sha256("claude-{MODEL}:{hash}")` but read back by prompt hash
alone, so changing model or effort silently returned the old configuration's
results. Errors are never cached — that would make a transient failure permanent.

### Observed behaviour

Against a ten-criterion rubric with planted defects, the stage correctly found
the inaccurate criterion (a 30-second requirement contradicting the user's
11 seconds, flagged both `inaccurate` and `counterproductive`), the
`vague_subjective`, `non_atomic`, `framing_double_negative`, `unnecessary`, and
`overlapping` plants, and the evidence guard rejected three unverifiable quotes.
Cost was $0.047 per call, about $0.90 for a 20-criterion task.

Two calibration notes worth carrying into a customer conversation. The auditor is
**aggressive on moderate issues** — it found a defensible moderate defect on
essentially every criterion, driving 240 to 100%. Before publishing rates, the
sensitivity should be calibrated against exemplars the customer has already
graded. And the gates count criteria, so a single harsh reading of a common
phrasing pattern moves the headline percentage a long way.

## The rating stage (phase 3)

The disagreement checks: 270 (per-criterion), 300 (per-dimension), 400 (the
side-by-side Likert). A 20-criterion task is 40 + 12 + 1 = 53 calls, so
`--dry-run` prints the call count before anything is spent, and any task already
found unauditable is skipped entirely.

### Blind and informed judgments cannot share a pass

These three checks measure distance from an independent rating, so the
contributor's values must be absent. But checks 280, 310, and 450 judge the
contributor's *own* citations and justifications, so they require exactly that
material. Putting them in one call, or even one ordering, would let the
contributor's cited turns anchor the independent rating — the failure mode that
makes a broken audit produce plausible numbers.

So the rating stage is a **blind pass only**. `assert_blind` re-checks every
prompt at runtime using the contributor's justification text as a canary: it is
long and distinctive, so it cannot appear by chance, and any code path that
starts pulling in contributor context will drag it along. A leak raises rather
than warns. Blind prompts also never use the word "contributor" at all, so no
framing can hint that a prior rating exists.

### Abstention is a first-class answer

Every schema offers `cannot_determine`, and the prompts say plainly that
abstaining is correct and useful. This exists because of artifact blindness: a
criterion about a video's resolution cannot be judged from a transcript that
records only that the video was delivered.

Counting an abstention as agreement would hide the blindness and understate
disagreement. Counting it as disagreement would invent defects. So it **leaves
the denominator** and the count is reported in `measurement.counts.abstained`.

Excluding abstentions makes the 20% rate threshold more sensitive, which is
deliberate — a fifth of what *could* be checked being wrong is the right reading
— but the base must not shrink without limit. Below
`Policy.min_judged_ratio` (default 50%) the check returns `not_evaluated` with
the judged ratio, rather than publishing a rate describing only whichever
criteria happened to be text-only.

### Conversation context

`context.py` renders turns with their 1-based exchange numbers (the same
convention checks 280 and 310 compare citations against), truncates
middle-out at a per-turn and per-conversation budget, and splits the comparison
budget evenly so a long Model A cannot crowd out Model B and bias the Likert.

Turns that delivered an artifact are **labelled as such**, detected by media
player chrome (`0:00 / 0:10`), "your X is ready" phrasing, and the
`artifacts_mentioned` field. Attachment filenames are listed with their contents
explicitly marked unavailable. The point is to tell the auditor what it cannot
see, so it abstains instead of guessing.

### Observed behaviour on the real conversation

Run against the real 16-exchange Gemini share link with a planted ten-criterion
rubric, half text-auditable and half about the video itself:

| criterion kind | judged | abstained |
| --- | --- | --- |
| text-auditable (5) | 5 | 0 |
| artifact-dependent (5) | 1 | 4 |

The separation is what the design intends, and the two interesting cases are the
exceptions. The artifact criterion it *did* judge was "between 10 and 12 seconds
long", read off the player timestamp `0:00 / 0:10` at medium confidence — the
duration genuinely is in the transcript. And on 4K resolution it abstained with
the right reason: the model asserted 4K in fifteen separate turns, but *"these
are only claims about intended specifications, not verification of what was
actually rendered."*

The three disagreements it found on text criteria were real and grounded: the
model claimed to have applied script updates it had not (the render came back
with the RPM counter dropping to 340 after the model confirmed it was locked at
9,750), and the user consequently had to restate a turn-1 requirement at turn 13.

Cost was $0.97 for 16 calls, so roughly $3.20 for a full 53-call task. Judged
ratio was 0.6, just above the 0.5 floor — a task with a more artifact-heavy
rubric would correctly return `not_evaluated` instead.

The residual limitation is not fixable in software: the contributor claimed all
ten criteria passed, and four of those claims are simply unknowable from a
transcript. The audit now says so explicitly rather than scoring them.

## Known spec gaps still open

1. **The spec's Likert buckets contradict the product.** Tab 1 (rows 135, 137)
   buckets the 1–7 preference `(1,2)` favors A, `(3,4,5)` neutral, `(6,7)` favors
   B. Across all 73 graded tasks the tool records 1–3 as "A wins" and 5–7 as "B
   wins", and never emits 4 at all:

   | Likert | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
   |---|---|---|---|---|---|---|---|
   | `responseIdx` | A | A | A | — | B | B | B |
   | tasks | 6 | 8 | 6 | 0 | 21 | 20 | 12 |

   So the midpoint is unused and the real boundary sits between 3 and 5, not
   between 2 and 6. Taking the spec at its word makes 27 of 73 tasks "neutral",
   and a neutral preference can never contradict anything, so check 460 goes
   silent on more than a third of the graded set. `likert_direction` therefore
   reads the recorded `responseIdx` whenever it is present and falls back to the
   spec's buckets only when it is absent.
2. **Weights outside 1–5 exist.** Check 260 states the scale is 1–5 with no
   negatives. Live rubrics carry 25 criteria across 14 tasks at −3, −4 and −5.
   Notably the audit workflow tab's issue table, whose `[weight -5]` reference
   was dismissed here as boilerplate from an unrelated project, may have been
   describing this after all. They are ingested rather than dropped, reported by
   preflight, and counted by 260 as two-level disagreements, because the
   auditor's weight is always in scale and there is no honest distance short of
   the maximum.
3. **Dimension ratings cover 7 dimensions; check 300 audits 6.** Contributors
   rate `collaboration_quality` on the same 1–5 scale as the rest, and nothing
   audits it. It is ingested and carried so the data survives intact, but no gate
   scores it: inventing one the spec never defined would publish a rate nobody
   agreed to. Safety is separate again — a `Yes`/`No` concern flag with a
   category and justification, never a 1–5 score, so it has no rating to disagree
   with.
4. **The L2 taxonomy has been revised at least once.** 32 live criteria carry
   leaves no revision we hold defines (`clarification timing`, `tool action
   execution`, `permission & scope boundaries`). Four more are renames of leaves
   we do hold, and are aliased only because each arrives carrying the canonical
   leaf's byte-identical description — `verify_alias_descriptions` re-checks that
   rather than trusting the name. Criteria whose leaf cannot be placed are
   skipped by check 210 instead of scored against a list the contributor was
   never offered.
3. **Artifact blindness** — share pages render deliverables as players and
   previews with no retrievable file. Contributor-uploaded attachments are the
   only path to auditing a produced artifact, so `attachments` is load-bearing
   for checks 95, 270, and 300. Confirmed on the real share page: every model
   turn that delivered a video renders as the eight-token string
   `Your video is ready! 0:00 / 0:10`. The deliverable is not in the transcript
   at all, which is why the user prompts carry nearly all the auditable text.
   There is now a way out of this that is not yet built: the form's
   `attachment_modela/b[].s3Url` are plain public HTTPS URLs, so the produced
   artifact can be downloaded and probed. Criteria the blind pass currently
   abstains on — "renders at 3840×2160", "runs 11 seconds" — become
   deterministic once `ffprobe` can see the file.
4. **A target deliverable absent from the conversation** has no home among the 21
   dimensions. The customer treats it as a bad trajectory; the nearest fit is the
   coverage pass, which only catches the rubric failing to reflect it.
5. **Coverage rates can exceed 100%.** Missing criteria raise the numerator
   without raising the denominator, which is what the spec asks for and what
   makes coverage failures bite — but a rubric where nearly every criterion is
   defective *and* something is missing produces "11 of 10". The band is correct
   either way; the rate is not publishable. Default behaviour follows the spec and
   flags the overflow in `measurement.counts`;
   `Policy(coverage_in_denominator=True)` counts absent criteria in both terms
   instead. The two disagree at the boundary — 10 clean criteria with one missing
   critical is 1/10 (fails 230) or 1/11 (does not) — so the customer should pick.

## Layout

```
config.py       Policy: every ambiguous value, versioned and reported
taxonomies.py   L1 labels, rating dimensions, issue severity census
errors.py       Verbatim error-code table (33 entries, 31 unique)
registry.py     The 21 dimensions, declarative
scoring.py      Bands, shapes, verdict envelope
models.py       Input contract
findings.py     What model stages emit and gates consume
preflight.py    Adapter plus hard-fail/warn validation
links.py        Check 90
gates.py        Census and all threshold arithmetic
rollup.py       Task verdict, output validation, batch metrics
transcripts.py  Transcript model, normalization, shingle containment (pure)
provenance.py   Link-vs-PDF comparison, verdicts, reuse detection (pure)
sources.py      PDF extraction and headless-Chrome rendering (all IO, fails soft)
verify.py       Runs provenance and maps it onto checks 90 and 1000
prompts.py      Prompt construction plus the independence guard
llm.py          ModelClient protocol, Claude CLI adapter, fake, retry, concurrency
cache.py        SQLite response cache with one shared key function
stages.py       Rubric stage: parse, validate, and run into the gates
context.py      Conversation rendering, artifact detection, evidence profile
rating_prompts.py  Blind prompts for 270/300/400 plus the blindness guard
rating_stages.py   Rating stage: abstention accounting and the judged-ratio floor
cli.py          Batch runner
dump_prompts.py Regenerates PROMPTS.md from the real builders
```

The fraud decision is pure and heavily tested; every network and file operation
is isolated in `sources.py` behind injectable callables, so the test suite runs
the entire provenance path with no browser and no fixtures on disk.

## Next

The natural phase 4 is the **informed pass**: checks 280 and 310 (are the
contributor's cited turns relevant?), 450 (are the justifications accurate and
evidenced?), and 460's escape clause (does the justification explain an
inversion?). These are the mirror image of phase 3 — they require the
contributor's own work in context, which is why they must run strictly after the
blind pass and never share a call with it. The gate arithmetic for all of them is
already built and tested; what they need is prompts and a stage.

Then 80, 95, 110, 220 (which the spec requires to carry a mandatory
second-opinion pass), 470, and 1000's judgment path.

## Reading tasks out of Snowflake

`ingest.py` builds `Task` objects from `public.taskattempts` rows. Pull the
latest attempt per task with `ingest.LATEST_ATTEMPT_SQL`, then:

```bash
~/.cursor/skills/redash/scripts/redash run -f latest.sql -o tasks.csv
python3 -m honeybee_qc.cli tasks.csv --from-snowflake --out report.json
```

Note the Redash data source: 22 is view-only for some accounts and Redash answers
`403` on query creation there. 30 (`_Snowflake (GenAI Ops)`) is writable.

**Read by shape, not by step ID.** The documented step map is a good starting
point and a bad contract. On a 25-task sample:

- The step documented as "B scores" carried both `response-a` and `response-b`,
  while the step documented as "A scores" was absent. Reading by ID returned
  zero ratings for product A on 14 of 25 tasks, silently halving check 270's
  denominator. Scores are therefore found by scanning every
  `RubricCriteriaRating` step and classifying entries by shape: a score entry
  carries `score`, a turn attribution carries `title` and no `score`.
- The trajectory field is not stably named. Product B alone appeared as
  `claude_trajectory_b`, `claude_trajectory`, `gemini_final_trajectory_b`,
  `gpt_trajectory` and `gpt_trajectory_b`. Product A is not always Gemini
  either, contrary to the skill: four tasks record `claude_trajectory_a`.
- A field named for one vendor may hold another's URL, and a "final trajectory"
  field may hold the whole newline-separated turn list. The product is inferred
  from the link's domain rather than the field name, and the last URL in a
  multi-URL field is the final one.
- The product-name prefillers are absent on roughly half the tasks.

Two further shape differences the contract has to absorb:

- `criterion_category` is one lowercased string, `"<l1>-<l2>: <l2 description>"`,
  not the two fields this contract carries. `taxonomies.parse_criterion_category`
  splits it, matching the closed L1 set as a prefix because one leaf
  ("Anti-Hallucination") contains a hyphen of its own, and folding the `&`/`and`
  and bare-L1 variants that appear live. It places 99.3% of 4,643 real criteria.
- Two `RubricCriteriaBuilder` steps exist. Only
  `step-RubricCriteriaBuilder-rubric01` is weighted and scored;
  `step-1784053126103-0h5ldq` is the unweighted Target Outcome Deliverables
  checklist, which maps to `Task.target_outcome`.

A task ingested this way carries no conversation: turn text lives on the share
pages and is fetched separately by `provenance.py`. Preflight warns rather than
rejecting, since only the blind rating pass needs it and that already returns
`not_evaluated` when evidence is thin.

The workflow also runs three Claude autoraters of its own, whose system prompts
sit at `dataSourceResults["before:0:data-source-systemPrompt-*"]`: a prompt
quality gate, a target-outcome coverage audit, and a rubric category audit. The
second overlaps this audit's coverage pass and the third overlaps checks 200/210,
so they are worth diffing against ours before treating either as authoritative.
