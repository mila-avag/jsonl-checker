"""Policy configuration for the HoneyBee_v2 QC audit.

Every value the spec leaves ambiguous lives here as an explicit, versioned field
rather than a literal buried in a gate. `Policy.provenance()` renders the active
choices so any published number can be reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

from .findings import DEFAULT_TARGET_OUTCOME_DEFECTS

POLICY_VERSION = "honeybee-v2-policy-1"

WeightBucket = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class WeightBucketMap:
    """Maps the 1-5 weight scale onto the high/medium/low levels check 260 counts.

    Grounded in the customer's own weight guide: 1 "minor positive",
    2 "moderately important", 3 "important", 4 "highly important", 5 "essential".

    The scale itself is settled: check 260's definition states "Criteria weights
    should be on a scale of 1 to 5 (no negative weights)". The audit workflow tab's
    issue table refers to weights 8-10, 4-7 and "[weight -5]", but that table is
    shared across QC projects (its examples are all tax and options trading), so the
    project-specific statement wins. What the spec still does not state is which of
    1-5 counts as high, medium, or low, which is the mapping below.
    """

    low: tuple[int, ...] = (1, 2)
    medium: tuple[int, ...] = (3,)
    high: tuple[int, ...] = (4, 5)

    def in_scale(self, weight: int) -> bool:
        return weight in self.low or weight in self.medium or weight in self.high

    def bucket(self, weight: int) -> WeightBucket:
        if weight in self.low:
            return "low"
        if weight in self.medium:
            return "medium"
        if weight in self.high:
            return "high"
        raise ValueError(f"weight {weight!r} is outside the 1-5 scale")

    def level_index(self, weight: int) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.bucket(weight)]

    def level_distance(self, cb_weight: int, auditor_weight: int) -> int:
        """Levels between two weights, treating an out-of-scale weight as maximally
        wrong rather than raising.

        Live rubrics do carry weights outside 1-5 -- 25 criteria across 14 tasks
        run from -3 to -5, against a spec that says the scale is 1-5 with no
        negatives. The auditor's weight is always in scale, so there is no bucket
        to compare against and no honest distance short of the maximum.
        """
        if not self.in_scale(cb_weight) or not self.in_scale(auditor_weight):
            return 2
        return abs(self.level_index(cb_weight) - self.level_index(auditor_weight))


@dataclass(frozen=True)
class Policy:
    # ---- scales -------------------------------------------------------------
    weight_scale: tuple[int, int] = (1, 5)
    weight_buckets: WeightBucketMap = field(default_factory=WeightBucketMap)

    # Spec gap 3, now closed: "differ by 3 or more points" (check 300) needs a
    # scale, and the spec never gives one. The authoring form does -- every
    # `<dim>_<a|b>` field is `number 1-5` (honeybee-l1-task-browsing reference.md).
    # This was 1-10 on inference before that landed, which halved the apparent
    # severity of every disagreement: a 3-point gap is 50% of this range, not 22%.
    dimension_rating_scale: tuple[int, int] = (1, 5)

    # Preference encoding. Two mutually exclusive forms are in play and the
    # conflict is not ours to resolve silently:
    #
    #  * bipolar_7      -- the QC spec's own form (tab 1 rows 135/137), a single
    #                      1-7 Likert whose value carries direction, bucketed
    #                      (1,2) favors A / (3,4,5) neutral / (6,7) favors B.
    #  * direction_magnitude -- what the authoring form actually stores: a
    #                      `responseIdx` (0=A, 1=B) naming the winner alongside a
    #                      `preferenceLikert` of 1-5 carrying only the margin.
    #
    # Both are normalised onto one signed axis (negative favors A, positive favors
    # B) so checks 400 and 460 are written once. The spec form is the default
    # because the spec is what this audit is contracted against; flip this when
    # reading tasks straight out of Snowflake.
    preference_encoding: Literal["bipolar_7", "direction_magnitude"] = "bipolar_7"

    likert_scale: tuple[int, int] = (1, 7)
    # Tab 1 buckets: (1,2) favors Model A, (3,4,5) neutral, (6,7) favors Model B.
    likert_favors_a: tuple[int, ...] = (1, 2)
    likert_neutral: tuple[int, ...] = (3, 4, 5)
    likert_favors_b: tuple[int, ...] = (6, 7)

    # Margin scale for the direction_magnitude form. There is no neutral bucket:
    # responseIdx always names a winner, so a recorded preference always has a side.
    preference_margin_scale: tuple[int, int] = (1, 5)

    # Criterion ratings are binary pass/fail; 280 keys off the score-0 set.
    criterion_rating_values: tuple[int, ...] = (0, 1)

    # Check 96, from the audit workflow tab's step 3: "Both the models require at
    # least 7 turns." Tab 2 gives the requirement no row, so this is the only
    # place the number is stated; `preflight.MIN_EXPECTED_EXCHANGES` reads it.
    min_turns_per_model: int = 7

    # A conversation we could not fetch in full is our shortfall, not the
    # contributor's, so a submission whose own citations run past the last turn
    # we retrieved is set aside rather than counted short. Off makes 96 gate on
    # raw hydrated counts, which manufactures a fail out of every login wall.
    turn_count_requires_complete_conversation: bool = True

    # ---- repeated sampling --------------------------------------------------
    # How many independent judgments to draw per model call, by check.
    #
    # A single sample of a noisy judge was deciding whether a task failed. Running
    # the identical check-280 prompt four times against the same submission
    # rejected 2, 1, 2 and 1 of the same 10 turn citations, and not one citation
    # was rejected by all four runs: every finding that pass produced was one a
    # rerun disagreed with. A gate fed one sample of that is not measuring the
    # contributor.
    #
    # Sampling is per check rather than global because it is the user's money.
    # The rubric stage alone is 509 criteria and roughly $28 a pass, so three
    # samples of everything would be about $85 a run. The default therefore draws
    # repeats only where the noise is demonstrated or the check is a known
    # over-flagger, and leaves everything else at one sample, which is byte-for-byte
    # the old behaviour and the old cost.
    #
    # An even count is deliberately allowed but is the worse buy: aggregation takes
    # the low median, so 2 samples means both must agree and the second sample buys
    # strictness rather than accuracy. Odd counts are what majority agreement is
    # for. Only checks listed in `sampling.SUPPORTED_SAMPLED_CHECKS` have an
    # aggregator; asking for repeats of any other check raises rather than silently
    # paying for samples nothing would combine.
    #
    # 110 is the one entry here with no remaining measured noise. It earned its
    # place by flagging 16 of 17 tasks against 14 false positives and no true
    # positives on the human sheet, but once the abstention channel landed, nine
    # fresh draws on each of three of those tasks came back clean 27 times out of
    # 27. With a base rate of zero there is no variance left for a repeat to
    # remove, so the three draws are a $3-a-run regression guard on a check with no
    # fail band rather than a measurement. Drop it with `--samples 110=1` once the
    # abstention fix has a full run behind it.
    samples_by_check: dict[int, int] = field(
        default_factory=lambda: {110: 3, 280: 3, 310: 3, 450: 3}
    )

    # Strict majority. A claim carried by exactly half the samples is not kept:
    # with the judge split down the middle the honest reading is that the item is
    # marginal, not that it is defective.
    sample_majority_ratio: float = 0.5

    # A judge that splits 2-1 on an item is telling us the item is marginal, and
    # that is information the report should carry rather than discard. Off only for
    # tests that assert on a bare finding list.
    record_sample_disagreement: bool = True

    # ---- band scoring -------------------------------------------------------
    # Scores 2 and 4 are deliberately never emitted. This is a decision, not a
    # spec gap: general grading instruction 2 supplies a rule for both
    # distinctions -- "when deciding between a 1 or 2, select a 1 if the attempter
    # put little to no effort" and "when deciding between a 3 or 4, use your best
    # judgement on how serious you think the minor issue affects the quality of
    # the task".
    #
    # Both rules turn on something no check here measures: how much effort the
    # attempter put in, and how badly a minor issue hurts the task overall. Every
    # check produces a band (fail / non_fail / clean), so a 2 or a 4 would be an
    # opinion published in the same field as a measurement. We emit the low end of
    # each band instead, which is the score the band alone can justify. Anyone
    # wanting the finer grade has the per-check evidence to make the call by hand.
    non_fail_score: int = 3
    fail_score: int = 1
    clean_score: int = 5

    # ---- gate thresholds ----------------------------------------------------
    l1_fail_rate: float = 0.30          # check 200, inclusive
    rubric_major_fail_rate: float = 0.10        # check 230, inclusive
    rubric_major_moderate_fail_rate: float = 0.15  # check 240, inclusive
    rubric_all_fail_rate: float = 0.20          # check 250, inclusive

    # Check 260, both inclusive, and both left at the spec's own percentages
    # ("at least 5% ... off by two levels", "at least 30% ... off by one level or
    # two"). What was wrong with this check was the numerator, not the percentage:
    # it counted every criterion where the auditor's point weight differed from the
    # contributor's, which on the first live run was 53% of criteria at the median
    # and flagged 17 of 17 tasks against a handful of human QC citations. The
    # numerator now counts only weights outside the range the customer's weight
    # definitions can support, so a rate at these thresholds means what the spec
    # says it means and the contractual numbers can stand unchanged.
    weights_two_level_fail_rate: float = 0.05
    weights_any_level_fail_rate: float = 0.30

    # Check 270, both inclusive and both counted in criteria, which is the unit
    # the spec's fail cell uses for both legs: "You disagree with the
    # contributor's ratings of 5 or more criteria, or 20% or more of the
    # criteria". A criterion the auditor rates differently on both models is one
    # disagreeing criterion. This was a judgment count (criteria x models) on
    # both sides of the rate, which halved it against the spec's denominator and
    # left 270 unable to fire on the rate leg until twice as many criteria were
    # wrong as the customer contracted for.
    rubric_eval_fail_count: int = 5
    rubric_eval_fail_rate: float = 0.20

    relevant_turns_fail_count: int = 3          # checks 280 and 310, inclusive

    dimension_major_delta: int = 3              # check 300, inclusive
    dimension_fail_major_count: int = 2         # check 300, inclusive
    dimension_fail_total_count: int = 5         # check 300, inclusive

    likert_fail_delta: int = 3                  # check 400, inclusive

    # Check 450, inclusive. THIS NUMBER IS OURS, NOT THE CUSTOMER'S. Every
    # threshold above is quoted from the spec; this one is not, because the spec
    # states none.
    #
    # What the spec does fix is the per-condition floors *inside* one
    # justification -- "at least 1 supporting claim does not logically defend the
    # verdict", two claims lacking evidence, two misconstrued quotes -- and those
    # are untouched in `JustificationFinding.triggered_conditions`. They are
    # statements about a single argument, which is why `justification_scope`
    # keeps them per-justification. The spec then says nothing at all about how
    # many of the roughly fifteen justifications a task carries have to trip
    # before the task fails, and this build read that silence as "one", so a
    # contributor who wrote fourteen sound arguments and one weak one failed on
    # the same code as one who wrote fifteen weak ones. On the first live run
    # that fired on 17 of 17 tasks with per-justification trip rates from 7% to
    # 80%, and a verdict that is always fail carries no information.
    #
    # 0.30 is the higher of the two proportions the spec itself states for "a
    # share of many small authored items is defective": 200's "at least 30% of
    # the labels are incorrect" and 250's 20% of criteria. The higher one is the
    # defensible choice here because 450's numerator is the most sensitive in the
    # audit -- one inaccurate load-bearing claim, or one fabricated quote, is
    # enough to put a whole justification in it -- so pairing it with the lower
    # share would make the task-level bar stricter than any proportional gate the
    # customer actually wrote while resting on a twitchier count than any of them.
    # At 30% of fifteen, five justifications must fail, which is a contributor
    # whose arguments break down across a third of the dimensions they wrote
    # about: that is the systematic defect "[Fail - Bad Justifications]" names.
    #
    # Applies only under per_justification scope. Pooling collapses the task to a
    # single synthetic justification, so there is no population left to take a
    # share of and the pooled branch keeps its any-trigger fail.
    justification_fail_rate: float = 0.30

    # ---- ambiguity resolutions (spec section 11) ----------------------------
    # Gap 6: 240 reads "between 1% and 15%" while 230/250 read "more than 0%".
    # A literal 1% floor leaves a single issue in a small rubric unrepresentable.
    non_fail_requires_nonzero_rate_only: bool = True

    # Gap 5 is closed and deliberately has no flag left: see check 270 above. A
    # switch back to a judgment denominator could only reintroduce a rate the
    # spec's "20% or more of the criteria" does not describe.

    # Gap 7: the spec counts incorrectly selected turns, not absent ones.
    count_missing_turns_as_incorrect: bool = True

    # Gap 1, now closed: the L2 taxonomy was missing from the spec export and
    # check 210 returned not_evaluated rather than a speculative clean. The
    # authoring form's own taxonomy supplies it (honeybee-l1-task-browsing
    # reference.md, from data-source-form-4607d2da02de).
    l2_labels_configured: bool = True

    # Missing criteria raise the numerator without raising the denominator, which
    # is what makes coverage failures bite -- but it lets the rate exceed 100%
    # when nearly every criterion is defective and something is also missing.
    # Spec behaviour is the default; the alternative counts absent criteria in
    # both terms, keeping the rate in range at the cost of diluting each gap.
    coverage_in_denominator: bool = False

    # Check 80: which classifications of a target-outcome entry count as a defect.
    #
    # Tab 2's non-fail cell is the whole definition of this check: "The target
    # outcomes list is misaligned with the prompt(s); it's incorrect, lists
    # outcomes which are not necessary or expected by the prompt(s), or
    # contradicts one or more prompts (except in cases where a requirement is put
    # forth but is changed/removed in later prompts -- the target outcome should
    # align with the expected final state)." Every clause there is a predicate on
    # an entry the contributor *listed*. Nothing in the cell, and nothing in the
    # audit workflow tab's step 6 ("Evaluate the Target Outcome List / Determine
    # if the target outcomes listed are correct and align with the prompt(s)"),
    # asks the auditor to derive the requirement set and report what the list
    # leaves out. The spec knows how to demand exhaustiveness when it wants it --
    # step 7 says the rubric "must be complete and exhaustive" and tab 2 gives the
    # rubric two whole error tiers for Missing Critical and Missing Non-Critical
    # criteria -- and it asks for none of that here. So the judge is never asked
    # to identify missing entries in the first place.
    #
    # `not_required` is the harder call and it is deliberately a policy question
    # rather than a hardcoded one. The spec's "lists outcomes which are not
    # necessary or expected by the prompt(s)" plainly does name that category, so
    # it is in scope in principle. What is out of scope is how the judge currently
    # decides it: it sees the prompts and nothing else, so on the first live run it
    # returned `not_required` for entries specifying the *content* of a deliverable
    # -- values drawn from input artifacts it was never shown, and guardrails like
    # "must not expose PII" -- with reasoning of the form "not traceable to any
    # user prompt". Tab 2's own definition column forbids exactly that inference:
    # the list "lists the necessary components for a successful response ... can
    # include components of artifacts and/or final model responses" and "may
    # include additional components beyond those required by the initial prompt".
    # An entry the prompts do not literally demand is therefore not yet evidence of
    # anything. Counting it anyway supplied 31 defects across 17 tasks.
    #
    # Default is the one classification the judge can decide on the evidence it
    # holds. Re-add "not_required" once the prompt carries tab 2's carve-out and
    # the channel has been re-measured. Flipping this tuple is the whole change:
    # the gate reads it and no other code encodes the scope.
    target_outcome_defect_classifications: tuple[str, ...] = DEFAULT_TARGET_OUTCOME_DEFECTS

    # A weight just outside the defensible range but inside one bucket (4 where the
    # definitions only support 5) is zero levels off, so it cannot reach a fail, but
    # it is still a weight the definitions do not support. Set False to let the
    # level thresholds alone decide the band.
    #
    # This replaces a flag that put the task in the non-fail band whenever the
    # auditor's own preferred weight differed from the contributor's at all, which
    # denied a clean to every task in the first live run: 348 of 509 criteria drew a
    # different point weight from the auditor, most of them one step down the same
    # scale. Disagreeing with the auditor's taste is not a defect; sitting outside
    # the defensible range is.
    weights_non_fail_on_out_of_band: bool = True

    # Check 300: contributor marked a dimension N/A where the auditor thinks it
    # applies (or vice versa). Reported as its own condition, never as a numeric
    # delta, and by default not counted as a defect at all.
    #
    # The spec says so twice. General grading instruction 4: "For SLA projects, if
    # a field is not relevant to your claimed attempt because it is not shown in
    # the UI or does not apply to the task type, choose the 'no issues' option."
    # And tab 1's pass criterion for each rating dimension reads "Your rating
    # matches the contributor's OR the contributor marks 'not applicable' where
    # the response cannot be assessed or vice versa" -- the applicability
    # divergence is inside the clean cell, on both sides of it. Counting it as a
    # minor issue manufactured a defect out of a dimension carrying no numeric
    # disagreement whatsoever, and could push a task to non_fail on that alone.
    #
    # The count stays published (`na_mismatches` in check 300's counts) because the
    # divergence is real signal about whether the two sides read the task type the
    # same way. Only move this off "ignored" if the customer says an applicability
    # divergence is a defect after all.
    na_mismatch_counts_as: Literal["major", "minor", "ignored"] = "ignored"

    # Check 450: thresholds such as "2 or more claims lack evidence" are counted
    # within a single justification, not pooled across the task.
    #
    # Tab 2 opens with "count the justification issues across all dimensions and
    # models and the ranking justification", which reads as pooled. We keep
    # per_justification anyway, because pooling is strictly harsher rather than
    # kinder: a pooled sum is never below the largest single justification's
    # count, so every task that fails per-justification also fails pooled, and
    # tasks with one bare claim in each of two different dimensions fail only
    # under pooling. Pooling would also make the two-claim thresholds meaningless
    # as written -- "2 or more claims lack evidence" is a statement about one
    # argument being unsupported, not about a task-wide tally that fifteen
    # justifications reach almost by construction.
    justification_scope: Literal["per_justification", "pooled"] = "per_justification"

    # Check 460: direction implied by the per-dimension ratings. `dominance`
    # only calls a direction when one model wins somewhere and loses nowhere,
    # which is the reading the spec's "support the opposite conclusion" implies
    # and the one auditors apply in their written rationales.
    ranking_direction_basis: Literal["dominance", "plain_mean", "weighted_mean"] = (
        "dominance"
    )

    # A dimension left not_evaluated (210 today) should not deny a task "clean".
    not_evaluated_blocks_clean: bool = False

    # ---- link handling ------------------------------------------------------
    verify_links_over_http: bool = False
    require_per_turn_links: bool = False

    # ---- provenance: does the live link match the uploaded PDF? -------------
    # Off by default because it needs a browser; the audit degrades to trusting
    # the PDF, which is exactly the hole this closes.
    verify_provenance: bool = False
    fetch_timeout_s: int = 60
    snapshot_dir: str = "snapshots"

    # Off by default so unit tests never open a real socket: `render_conversation`
    # only reads a `final_outputs_*` attachment's actual bytes when this is set,
    # which the CLI does whenever `--fetch-conversations` is passed -- the same
    # gate that already governs every other network read in a prompt build.
    fetch_attachment_text: bool = False

    # Transcripts run to tens of megabytes, so the download gets its own budget
    # rather than sharing the per-page render timeout.
    pdf_download_timeout_s: int = 180

    # Off by default: OCR costs roughly a second per page and only helps the
    # transcripts that were printed with their text converted to outlines.
    pdf_ocr_fallback: bool = False

    # On by default, unlike `pdf_ocr_fallback` above: a PDF has other backends to
    # try first and OCR is only a last resort, but an image attachment has no
    # text layer at all -- OCR is the *only* path to any text, so leaving it off
    # would make every image attachment read as empty by construction. This is
    # OCR (visible text extraction), not general image understanding: a photo,
    # a chart with no labels, or a diagram with no legible text still yields
    # nothing, because there is no vision model in this pipeline's call path.
    image_ocr_fallback: bool = True

    # Bounds a single Tesseract call so one oversized or pathological image
    # cannot stall a task's audit indefinitely; pytesseract enforces this by
    # killing the underlying process and raising, which degrades to "no text
    # available" exactly like any other OCR failure.
    image_ocr_timeout_s: int = 20

    # Check 95 matches a filename the model claimed against the upload
    # manifest's own name for the same file. An exact match misses the common
    # case where the two agree on everything but a suffix the upload path
    # added on its own -- a duplicate-upload " (2)", a "_v2" the contributor
    # typed, a client-added " - Copy" -- which is cosmetic, not a sign the
    # claimed file was never delivered. Below this similarity ratio on the
    # case-folded basename, treat it as actually missing rather than renamed.
    #
    # 0.90 was where the near-misses lived. On a manual audit of 9 check-95
    # fails, 5 were false and their similarity ratios clustered at 0.87-0.89 --
    # a capture-tool screenshot name against the contributor's own name for the
    # same image, a version suffix, a space that became an underscore -- while
    # every true fail scored zero similarity against every upload, because the
    # claimed file genuinely was not there. Two points of slack separates those
    # populations, and the LLM confirmation below is what stops the slack from
    # turning into a missed swap.
    filename_fuzzy_match_threshold: float = 0.87

    # Check 95, and the reason the threshold above can be loosened safely.
    #
    # String distance cannot see that `Screenshot 2026-08-02 at 11.14.31.png` and
    # `dashboard-after-fix.png` are the same image, that a fenced code block the
    # model printed inline is not a file it claimed to attach, or that
    # `report_final.pdf` and `report_final_v2.pdf` are two different documents at
    # 0.94 similarity. Every fail is therefore confirmed by one model call per
    # submission before it is published: the judge is shown the claimed name, the
    # sentence claiming it, and the full upload manifest, and decides per file
    # whether it is present under another name, was never a file at all, or is
    # genuinely missing. A pass costs nothing -- the call only happens where the
    # subtraction already found a shortfall.
    artifact_miss_requires_llm_confirmation: bool = True

    # A 5-word window matching is hard to achieve by coincidence, which is what
    # keeps honest topical overlap from reading as a match.
    provenance_shingle_size: int = 5
    provenance_min_turn_tokens: int = 8
    provenance_min_scored_turns: int = 3

    provenance_turn_match_threshold: float = 0.60
    provenance_verified_ratio: float = 0.80
    provenance_contradicted_ratio: float = 0.40
    provenance_reverse_min: float = 0.30

    # Every task is a distinct conversation, so the same share page or transcript
    # under two task IDs is copy-paste rather than coincidence.
    duplicate_across_tasks_is_unauditable: bool = True

    # ---- rating stage (checks 270, 300, 400) --------------------------------
    # These three were 60000 / 80000 / 6000, and at those values the render
    # budget was itself a source of disagreement. On a manual audit of 12
    # check-300 fails, 7 were false: the judge was handed the first 10-13 turns
    # of a 20-22 turn conversation, was not told the rest existed in any way it
    # weighted, and rated a dimension that the turns it never saw would have
    # settled. Check 400 was worse -- `render_comparison` halves
    # `max_comparison_chars`, so each side got 40000, less than a single
    # conversation got for check 300, and one sampled task showed Model B cut to
    # one visible exchange against Model A's eight.
    #
    # The numbers below are chosen so that truncation becomes the exception
    # rather than the median case: 160000 characters is roughly 40k tokens, which
    # covers every conversation in the batch-9 population at full length, and
    # 260000 keeps each side of the comparison (130000) above what a single
    # conversation used to get on its own. Raising them costs input tokens on
    # every rating call, which is the trade being made deliberately: a fail
    # produced by a render budget is a fail the customer cannot act on.
    max_conversation_chars: int = 160000
    max_comparison_chars: int = 260000
    max_turn_chars: int = 12000

    # When a submission's conversation carries no model replies -- hydration
    # failed, or the product only ever published a single final share link, which
    # is the norm for slot B -- the exported chat PDF is the other copy of the
    # same conversation. `ingest` already records it (`model_{a|b}_chat_download`
    # -> `ModelSubmission.transcript_pdf`) and nothing read it, so a submission
    # whose share page did not render reached the blind judges as a user-turn-only
    # skeleton and every judgment against it was either an abstention or a guess.
    # Read as an evidence block rather than as parsed turns: the PDF is a print of
    # the page, so its speaker structure is not recoverable reliably enough to
    # number turns off, and a turn number that disagrees with the contributor's
    # would break checks 280 and 310. Requires `fetch_attachment_text`, the same
    # flag that governs every other network read in a prompt build.
    transcript_pdf_fallback: bool = True

    # Check 300, and the reason this build no longer publishes a blind rater's
    # arithmetic as a defect.
    #
    # The check as specified is a delta between the contributor's 1-5 rating and
    # an independent one. Blindness is what makes that delta mean anything -- a
    # judge shown the contributor's number agrees with it -- but blindness also
    # means the judge is answering "what would I have put here", and two
    # defensible readings of the same conversation are three points apart often
    # enough that on the manual audit the delta was measuring the judge as much
    # as the contributor.
    #
    # So the blind rating still happens and is still reported, and it no longer
    # decides anything on its own. Every dimension the blind pass disagrees on
    # goes to a second, deliberately *informed* call that is shown the
    # contributor's rating and their own justification and asked one question:
    # is that rating defensible against this conversation? Only a disagreement
    # the informed pass calls indefensible counts toward 300's bands. The rest
    # are published as `blind_only`, which is the label the report needs: a fail
    # nobody can act on because our rater picked a different number is not the
    # same finding as a rating the transcript contradicts.
    #
    # Off restores the pre-adjudication behaviour (blind delta decides), which
    # is the spec read literally and is what the recorded batch-9 numbers were
    # produced under.
    dimension_disagreement_requires_informed_confirmation: bool = True

    # Check 400, same change for the same reason. One Likert call per task, so
    # the confirming call is at most one more.
    ranking_disagreement_requires_informed_confirmation: bool = True

    # A rating whose blind judgment was made from a truncated render is reported
    # as such and, by default, is not counted even when the informed pass agrees:
    # both passes read the same truncated conversation, so the confirmation is
    # not independent of the gap that caused the disagreement. Off counts them,
    # with the truncation still labelled in the output.
    exclude_truncated_evidence_disagreements: bool = True

    # An auditor that cannot see the deliverable must be able to say so. Counting
    # an abstention as agreement would hide the blindness and understate
    # disagreement; counting it as disagreement would invent defects. It leaves
    # the denominator instead, and the abstention count is reported.
    exclude_abstentions_from_denominator: bool = True

    # Below this share of judgeable items, a rate is not worth publishing: it
    # would be computed over whichever few criteria happened to be text-only.
    min_judged_ratio: float = 0.50

    def provenance(self) -> dict:
        d = asdict(self)
        d["policy_version"] = POLICY_VERSION
        return d


DEFAULT_POLICY = Policy()
