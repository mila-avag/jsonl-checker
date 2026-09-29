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
    """Maps a weight scale onto the high/medium/low levels check 260 counts.

    The default is the 1-5 scale, grounded in the customer's own weight guide:
    1 "minor positive", 2 "moderately important", 3 "important", 4 "highly
    important", 5 "essential". Check 260's L1 definition states "Criteria weights
    should be on a scale of 1 to 5 (no negative weights)". The audit workflow
    tab's issue table refers to weights 8-10, 4-7 and "[weight -5]", but that
    table is shared across QC projects (its examples are all tax and options
    trading), so the project-specific statement wins. What that spec still does
    not state is which of 1-5 counts as high, medium, or low, which is the
    mapping below.

    Not every project uses that scale, and getting it wrong is silent rather than
    loud. Honeybee Aspirational's spec states a three-point scale -- "+1 (low),
    +2 (medium), +3 (high)" -- and its rubrics bear that out: across 1,779
    criteria on 73 tasks the weights are 243 ones, 742 twos and 794 threes, with
    no 4 or 5 anywhere. Read through the 1-5 map, its most important criteria
    bucket as "medium" and its "high" level is unreachable, so nearly half the
    corpus is compared against the wrong rung and every off-by-one measurement
    is understated. `THREE_POINT_WEIGHTS` below is that project's map; a project
    profile names which one applies.
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


# Honeybee Aspirational's scale, quoted from its rubric: "+1 (low), +2 (medium),
# +3 (high)". One weight per level, so "off by one" and "off by one level" are
# the same statement here -- which is what makes the spec's own "off by 1" and
# "off by 2" thresholds directly measurable rather than needing a mapping.
THREE_POINT_WEIGHTS = WeightBucketMap(low=(1,), medium=(2,), high=(3,))


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
    # 450 draws five rather than three because three was measured as not enough.
    # Re-auditing sixteen tasks that had already been judged, with the prompt
    # substantively unchanged, moved their defect counts by 6->3, 7->11, 8->5 and
    # 7->5 -- and those are the numbers *after* three-draw majority voting, not raw
    # draws. The aggregate barely moved (63 issues to 60), so the judge is not
    # biased, it is noisy per item; and because tasks sit near the share threshold,
    # per-item noise lands directly on the task verdict. That is what made recall
    # estimates for this check range from 22% to 86% across samples of the same
    # population. Five draws needs three to agree instead of two, which is the
    # cheapest available brake on a single erratic draw carrying an item.
    samples_by_check: dict[int, int] = field(
        default_factory=lambda: {110: 3, 280: 3, 310: 3, 450: 5}
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

    # Checks whose fail threshold is exclusive rather than inclusive -- fail
    # strictly above the rate, not at it. Empty by default because the L1 spec
    # words every one of these gates as "at least". Honeybee Aspirational's V5
    # rubric words two of them the other way, and on a rate whose denominator is
    # a rubric of 15-30 criteria the boundary is a value tasks actually land on,
    # not a rounding curiosity.
    exclusive_fail_thresholds: frozenset[int] = frozenset()

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

    # Check 280 only, inclusive, and a floor on top of the count above rather
    # than a replacement for it: a task fails 280 when it carries at least
    # `relevant_turns_fail_count` incorrectly cited turns AND those turns are at
    # least this share of the criteria the check audited.
    #
    # The count leg alone is a defect *count* over a population that this corpus
    # runs from 0 to 54 criteria wide, and it behaves accordingly: the numerator
    # correlates with rubric size at +0.61, the median `items_audited` on a 280
    # escalation is 34 against 0 on a non-escalation, and 10 of the 14 false
    # fails attributed to the 280/310 family were 280. A task with a
    # 34-criterion rubric gets 34 chances to reach 3; a task with none can never
    # trip it at all. That is a rubric-size test wearing a defect-density test's
    # clothes.
    #
    # 310 deliberately has no rate floor. Its item population is always the
    # task's dimension ratings -- 14 items, every task -- so it has no size to
    # correlate with, and its measured correlation is -0.26. A shared floor
    # would raise 310's effective bar from 3 to 5 for no reason anyone has
    # measured.
    #
    # The absolute count is kept as well, deliberately: a small rubric with
    # three genuinely fabricated citations is a real defect, and a pure rate
    # would let it through. Set to 0.0 to restore the pure absolute count.
    rubric_relevant_turns_fail_rate: float = 0.33

    # Check 300, inclusive, and a COUNT of confirmed disagreements. 0 disables it,
    # which is the default: this leg is the one `dimension_fail_total_points`
    # below was measured to replace, and switching it back on globally would
    # reinstate the false fails that change removed. Honeybee Aspirational's V5
    # rubric states it outright ("You have 5 or more major or minor
    # disagreements"), so that project sets it, at a bar two and a half times the
    # count that was measured to be wrong.
    dimension_fail_total_count: int = 0

    dimension_major_delta: int = 3              # check 300, inclusive
    # Check 300, inclusive, and see the note on `dimension_fail_total_points`
    # below for why this leg cannot fire on its own at the current numbers.
    dimension_fail_major_count: int = 2         # check 300, inclusive
    # Check 300, inclusive, and a SUM OF POINTS rather than a count of items.
    # This replaces `dimension_fail_total_count = 2`, which counted dimensions
    # carrying any nonzero confirmed delta and so scored a 1-point disagreement
    # exactly like a 4-point one.
    #
    # The spec's own fail cell says "total point-deltas" and its number is 5. The
    # mistake was never the unit, it was the population: the spec's 5 was applied
    # to raw *blind* deltas, and applied there it fails 85.7% of a human-QC-clean
    # corpus. Since informed confirmation landed the numerator is
    # confirmed-indefensible gaps only, which is 7.6% to 9.3% of the blind gaps
    # on the two scored corpora. Counting the spec's unit over the confirmed
    # population, rather than counting items over it, is what this field does.
    #
    # 6 is measured on 171 tasks with every gap magnitude recovered from the
    # stored draws (zero misses) and the published band reproduced on all of
    # them:
    #
    #                                    91 QC-clean tasks   60 mixed   live 40
    #     n>=2 or majors>=2 (previous)      7 fails            6         19
    #     points>=6 or majors>=2 (this)     2 fails            1         10
    #
    # That removes 5 of the 7 false fails on the QC-clean corpus (71%), takes the
    # unambiguous ones -- QC scored the task 5 and cited no dimension-rating
    # issue -- from 3 to 1, and on the mixed corpus removes both false fails
    # while keeping the one fail QC also failed the task for.
    #
    # Recall cost, stated plainly: three of the four fails that sat on QC=Fail
    # tasks in the 60-task corpus become `non_fail`, including the one task where
    # QC explicitly cited a dimension-rating issue and 300 agreed -- its
    # confirmed gaps are [1,1,1]. At the *task verdict* level the cost is close
    # to zero, because all three still fail on 450 and/or 310, so no QC-failed
    # task becomes a pass; what changes is per-check attribution. That is the
    # right trade for a check nobody can use as a per-check defect rate while its
    # fail band fires on three 1-point disagreements.
    #
    # What is robust across every corpus and rule tried is the direction: 2
    # confirmed gaps of any magnitude is too low a bar, and the 1-point gap is
    # where the false fails live. The exact number 6 is not robust; it is priced
    # on 7 false fails and 1 QC-corroborated true positive. The next-best point,
    # if discarding 1-point recall is unacceptable, is a count leg at 3 (3 fails
    # on the QC-clean corpus, 4 of 7 false fails removed).
    #
    # THE MAJOR LEG CANNOT FIRE ON ITS OWN AT THESE NUMBERS, and that is worth
    # stating where a reader will see it, because the previous comment left the
    # same trap unmarked. `dimension_fail_major_count` (2) gaps of
    # `dimension_major_delta` (3) points is 6 points, which already trips this
    # leg, so every numeric disagreement the major leg would fail a task for is
    # one this leg fails it for first. It was dead code under the old count leg
    # for the mirror-image reason (two major gaps are already two total gaps) and
    # it is dead again here. It is kept rather than deleted for two reasons: it
    # is the leg that still binds if this threshold is ever raised above
    # `major_count * major_delta`, and it is the only route by which an
    # applicability divergence can fail a task when `na_mismatch_counts_as` is
    # set to "major", since an N/A mismatch carries no numeric delta and so
    # contributes no points. `evaluate_check_300` publishes
    # `major_leg_can_fire`, computed from these three numbers, so the condition
    # is visible in the output instead of resting on this comment.
    dimension_fail_total_points: int = 6

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
    # 0.20 is the lower of the two proportions the spec itself states for "a
    # share of many small authored items is defective": 200's "at least 30% of
    # the labels are incorrect" and 250's 20% of criteria. This sat at 0.30 on
    # the argument that 450's numerator is the most sensitive in the audit -- one
    # inaccurate load-bearing claim, or one fabricated quote, is enough to put a
    # whole justification in it -- so it deserved the more forgiving share. The
    # first blind batch measured that argument and it does not hold, for a reason
    # the argument overlooked: the twitchy cases are not on this leg at all.
    # `justification_severe_conditions` already fails a task on one fabricated
    # citation or one false load-bearing claim at any share, so the share rule
    # never was the thing standing between a single bad claim and a fail. Setting
    # it high did not buy protection from the sensitive numerator; it only raised
    # the bar for the accumulating defect this leg exists to catch.
    #
    # Scored against 117 human-labelled tasks, moving 0.30 -> 0.20 takes check
    # 450 from 14% recall at 43% precision to 55% recall at 52% precision, and
    # the whole audit from 52% recall at 31% precision to 61% at 34% -- three more
    # true fails for one more false one. 0.10 was also swept and is worse: one
    # further true fail for five false ones. Both numbers are in-sample on a
    # single batch and the labels behind them are unadjudicated single-auditor
    # opinions, so this is a better-supported choice than 0.30 was, not a
    # settled one.
    #
    # At 0.20 of fifteen, three justifications must fail. That is still a
    # contributor whose arguments break down on a fifth of the dimensions they
    # wrote about, which is the systematic defect "[Fail - Bad Justifications]"
    # names, and it is no longer stricter in effect than the customer's own
    # comparable gates.
    #
    # 0.20 -> 0.10 travels with `justification_draw_agreement` 3 -> 5 and is not
    # defensible without it. The paragraph above rejecting 0.10 was measured at a
    # 3-of-5 majority, where lowering the share admits the minority-vote items the
    # majority had already let through -- one further true fail for five false
    # ones. At unanimity those items are gone before the share is taken, so the
    # two knobs are not independent and the earlier sweep does not price this
    # pairing. Swept jointly, k=5 with 0.10 gives false fails 42 -> 38 at
    # unchanged task-level recall; k=3 with 0.10 gives 43 false fails. Move them
    # together or not at all.
    #
    # A caution for whoever tunes this next: at unanimity the share is nearly
    # inert. On the 60 landed `priority_batch` tasks, k=5 escalates 9 of 60 at
    # every share from 0.10 to 0.30, because almost nothing survives unanimity in
    # enough quantity to reach a share gate and
    # `justification_severe_conditions` carries what is left. Four of those nine
    # escalations bypass the share entirely. Tuning this number is not where the
    # remaining leverage on this check is.
    #
    # Applies only under per_justification scope. Pooling collapses the task to a
    # single synthetic justification, so there is no population left to take a
    # share of and the pooled branch keeps its any-trigger fail.
    # Which rule check 450's band comes from. "share_or_severe" is this build's,
    # measured and argued in the notes below: a share of justifications carrying
    # any condition, plus a one-is-enough escape for the severe ones.
    #
    # "all_models_or_all_comparisons" is Honeybee Aspirational's V5 rule, which
    # is structural instead -- every model, or every comparison, must carry a
    # major issue before the task fails. It is dramatically more forgiving: one
    # fabricated citation fails a task under the first rule and is a non-fail
    # under the second so long as another model's justifications are clean. Not a
    # threshold that can be tuned into agreement, which is why it is a named rule
    # rather than a number.
    justification_fail_rule: Literal[
        "share_or_severe", "all_models_or_all_comparisons"
    ] = "share_or_severe"

    justification_fail_rate: float = 0.10

    # Check 450 only: how many of `samples_by_check[450]` draws must report a
    # defect on one justification for it to survive aggregation. 3 is exactly the
    # `median_low` this replaced -- for five draws `median_low(v) >= t` holds iff
    # at least three draws have `v >= t` -- so 3 reproduces every run scored
    # before this field existed, and the sweep in
    # `audit_runs/false_fail_analysis/sweep450.py` confirms that end to end.
    #
    # 5 is unanimity, and it is set there because a bare majority of this judge is
    # close to a coin flip. Measured over 400 complete five-draw groups on the
    # 151-task sheet22 corpus: only 17% of flagged justification items were
    # flagged by all five draws and 56% were the opinion of one or two. That is
    # not a threshold problem -- 30 of the 32 tasks QC tagged [Fail - Bad
    # Justifications] have a non-zero numerator at one draw, so the judge sees the
    # defects; it does not see them consistently. `median_low` kept everything at
    # three of five, and combined with `justification_severe_conditions` failing a
    # task on one item, a 3-of-5 coin flip on one justification of fifteen failed
    # a $6 task.
    #
    # Out-of-sample check, because the number above was measured on the same
    # corpus the threshold was tuned on. On 60 tasks of the `priority_batch` run
    # -- patched build, never-reviewed population, no QC ground truth -- raw
    # unanimity is 23.6% and the minority rate 54.0%. The minority figure
    # replicates outright; the unanimity gap is a composition effect, since that
    # corpus is 77% defect-rich against sheet22's 41% and defect-rich tasks are
    # twice as unanimous in either corpus. Standardised to sheet22's mix it is
    # 18.3% against 17.2%. See `audit_runs/oos_validation/OOS_450_VALIDATION.md`.
    #
    # Priced in-sample at 5: false fails 42 -> 38 with task-level recall
    # unchanged at 31, and 450's own precision 48% -> 50% against recall 47% ->
    # 34%. The recall lost is the check's own, not the audit's -- every task 450
    # stops failing is either failed by another check or was a false fail. Priced
    # out-of-sample at 5: 450 escalates 16 of 60 tasks -> 9, seven tasks stop
    # failing it, all seven still fail another check, and no whole-task verdict
    # moves. k=4 at the same share instead *adds* a task-level fail, which is why
    # this is 5 and not the softer 4.
    #
    # Not an open question, and not a knob to reopen without new measurement: the
    # value is settled at 5 on both corpora. It stays a named field because the
    # severe route is still being measured separately and because 3 is what
    # replays a pre-patch run.
    #
    # Rescaled to however many draws came back, so a group that lost one is not
    # silently held to a different rule: see `sampling._justification_votes`.
    justification_draw_agreement: int = 5

    # Check 450: the conditions the share above does not govern. A justification
    # tripping one of these fails the task on its own, whatever its share of the
    # population.
    #
    # The share is a task-level judgment about a *pattern* -- arguments that
    # break down across a third of the dimensions -- and it is the right
    # instrument for the conditions whose own floors are soft: two unsupported
    # claims, two misread quotes, a weighting the auditor would have set
    # differently. It is the wrong instrument for the two the spec stamps "One is
    # enough to fail" in the auditor's own instructions (`inaccurate_evidence`
    # and `inaccurate_primary_claims`). A fabricated quote or a false
    # load-bearing claim is not evidence of a tendency that needs corroborating
    # across the other fourteen justifications; it is the defect itself, and the
    # spec set its floor at one precisely so that it would not need company.
    #
    # Read as a share, those two floors were unreachable: one instance of
    # fifteen is 6.7% against a 30% gate, so the strictest numerator in the audit
    # was the one the gate could never see. In the 53-task validation run this
    # cost five of the six tasks the human QC team failed on justification
    # grounds, including one that tripped `cites_incorrect_evidence` on 27% of
    # its justifications -- four separate fabricated citations -- and passed.
    #
    # `is_inaccurate` covers a second branch, the wrong subordinate claim, which
    # the spec does not call one-is-enough and this build now does. The spec's
    # equivalence was two subordinate errors to one load-bearing one, which is
    # the reason the two counts share a condition in the first place; the floor
    # that enforced it lives in `findings.SECONDARY_INACCURACY_FLOOR` and is set
    # to 1 there, with the argument for the departure beside it. Read this tuple
    # with that floor: the severe escape is now reachable from one subordinate
    # inaccuracy, so it is a wider door than the paragraphs below measured.
    #
    # A correction worth carrying, because an earlier analysis of this field drew
    # the wrong conclusion from the right observation. Setting this tuple to
    # ("cites_incorrect_evidence",) gives byte-identical results to emptying it on
    # both measured corpora, and that was read as the condition never firing. It
    # fires: 7 items on 3 of 151 sheet22 tasks and 12 items on 7 of 60
    # `priority_batch` tasks. What the identity proves is only that it was never
    # *decisive*, because every task carrying it also carried `is_inaccurate` or
    # cleared the share honestly. At `justification_draw_agreement = 5` that stops
    # being true: one escalation of 60 is carried by `cites_incorrect_evidence`
    # alone, the minority-vote `is_inaccurate` findings beside it having been
    # filtered out. The spec's own one-fabricated-quote condition is load-bearing
    # for the first time in either corpus, so it must not be treated as inert.
    #
    # Also measured, and now partly superseded: `is_inaccurate` reduced to
    # `inaccurate_primary_claims >= 1` in practice -- the two-secondary-claims
    # branch accounted for 0 of the 5 severe escapes out of sample, which is one
    # of the reasons its floor moved to 1 -- and 31% of
    # 450's escalations bypass the share through this tuple entirely, rising to 44%
    # once the agreement bar tightens. Whether the severe route should carry its
    # own agreement bar is a separate decision with its own measurement in flight.
    # It would go beside `justification_draw_agreement` rather than in here.
    #
    # Emptying this tuple restores the pure share gate, which is the supported
    # way to run without the escape. Conditions are named as
    # `JustificationFinding.triggered_conditions` emits them; a name with no
    # counterpart there is inert rather than an error, since the finding's
    # vocabulary is the spec's and not this field's to police.
    justification_severe_conditions: tuple[str, ...] = (
        "is_inaccurate",
        "cites_incorrect_evidence",
    )

    # Check 450: how many independent draws must have named a severe condition
    # before it is allowed to bypass the share. Counted out of the draws that came
    # back, not rescaled to `samples_by_check[450]`, which is the whole point of
    # the field.
    #
    # The escape has never been the free-for-all it reads as. It consults the
    # *aggregated* finding, so `sampling.aggregate_justifications` has always
    # bounded it: measured over the 151-task sheet22 corpus and the 60 landed
    # `priority_batch` tasks, every severe condition that survived aggregation on
    # a complete five-draw group was named by at least three draws, which is
    # exactly what `median_low` requires. A draw-agreement bar stated as a share
    # of the group is therefore inert at or below that, and 1 reproduces today.
    #
    # What the aggregator does not bound is a group that lost draws.
    # `aggregate_justifications` returns a lone draw verbatim when `total <= 1`,
    # and on sheet22 two tasks -- 6a7f7ca67216429b95c72149 and
    # 6a800beb4c1c25ccc73e4ab7 -- failed on a severe condition exactly one stored
    # draw ever named, on one justification of fifteen. Both survive the 5-of-5
    # aggregation change, because there is nothing for five-of-five to filter
    # when only one draw exists. 2 is the smallest value that closes that, and it
    # is free: on the 91-task all-QC-pass corpus and the 151-task corpus it moves
    # neither true nor false task-level fails, and on the live run it moves
    # nothing at all. It removes two of 450's 22 escalations at k=5, which are two
    # claims the check could not support.
    #
    # Above 2 it stops being free. 5 -- "the escape needs all five draws,
    # whatever came back" -- costs one true fail on sheet22 for no false-fail
    # gain, because it also rejects a group of four that agreed unanimously. The
    # spec's own reading is that a lost call is our failure and not the
    # contributor's, and 2 is the version of this that only denies an escape no
    # draw ever corroborated.
    #
    # Findings nothing sampled carry `draws_combined = 0` and are exempt rather
    # than denied: a synthesised empty-justification finding and a single-sample
    # run have no draw record, and refusing them the escape would make this field
    # a switch on `samples_by_check` instead of a bar on evidence.
    justification_severe_min_draws: int = 2

    # Check 450: the share of a task's justifications that must carry a severe
    # condition before one of them may fail the task on its own. 0.0 is off, and
    # off is today.
    #
    # This is the "one of many" question and it is a policy decision, not a
    # calibration. The spec stamps `inaccurate_evidence` and
    # `inaccurate_primary_claims` "One is enough to fail" in the auditor's own
    # instructions, so any value above 0 overrides the customer's own floor and
    # nobody should set it without saying so. It is a field rather than a patch so
    # that the decision can be taken deliberately, and measured before it is.
    #
    # Measured, with `justification_draw_agreement` at 5 and the share at 0.10:
    # every severe escape left on either corpus is one severe justification of
    # thirteen to fifteen, 6.7% to 7.7%, and all five draws agree on all of them.
    # So any floor above 7.7% closes the route outright on this data and the
    # choice is not between floors, it is whether to keep the route. At 0.10 it
    # takes the 151-task corpus from 38 false task-level fails to 36 for one true
    # fail, the 91-task corpus from 34 to 32, and flips one live verdict.
    #
    # A share rather than a count because the complaint is not that one is never
    # enough. One inaccurate load-bearing claim in eight justifications and one in
    # fifteen are different claims about a contributor, and a share says so where
    # a count of two says only that one is never enough. On this corpus the two
    # are indistinguishable -- a count of 2 gives the same numbers everywhere --
    # so the share is chosen on the argument and not on the measurement.
    justification_severe_min_share: float = 0.0

    # ---- ambiguity resolutions (spec section 11) ----------------------------
    # Gap 6: 240 reads "between 1% and 15%" while 230/250 read "more than 0%".
    # A literal 1% floor leaves a single issue in a small rubric unrepresentable.
    non_fail_requires_nonzero_rate_only: bool = True

    # Gap 5 is closed and deliberately has no flag left: see check 270 above. A
    # switch back to a judgment denominator could only reintroduce a rate the
    # spec's "20% or more of the criteria" does not describe.

    # Gap 7: the spec counts incorrectly selected turns, not absent ones.
    count_missing_turns_as_incorrect: bool = True

    # Check 95's structural leg. The customer fails a task whose final-outcome
    # upload section is empty even when every file was attached turn by turn,
    # and the named-file comparison cannot express that, because ingest unions
    # the two upload fields before the gate ever sees them. Off switches the leg
    # back off and leaves 95 a pure claimed-vs-uploaded comparison.
    final_outputs_section_required: bool = True

    # Check 95's extraction scope. "conversation" lists every file the model
    # produced across the whole session, which is the L1 reading. "final_response"
    # lists only files mentioned in the last assistant turn, which is V5's rule
    # verbatim: "every output file mentioned in the final response was uploaded",
    # with intermediate versions and earlier-turn files explicitly excused.
    artifact_scope: Literal["conversation", "final_response"] = "conversation"

    # Check 91 Unique Links. V5 fails at "4+ turns for Gemini that do not contain
    # a unique link" and non-fails at "1-3". L1 has no such dimension.
    unique_links_fail_count: int = 4

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

    # The adjudication pass reads the same conversations under a wider budget, and
    # the width is the whole point of it.
    #
    # Raising `max_conversation_chars` to 160000 did not stop truncation, because
    # the flag that suppressed a finding was mostly not about the conversation
    # budget at all: `max_turn_chars` elides the middle of any turn past 12000
    # characters, and live turns run to 73000 (one task) and 95000 (another). On
    # three of the four tasks whose check-300 gaps were dropped for truncation,
    # every turn had been rendered and the conversation was well inside 160000 --
    # a single long turn set the flag, and the check published clean.
    #
    # These budgets exist because the blind pass and the adjudication pass have
    # different economics. The blind pass makes twelve dimension calls per task and
    # pays the conversation once per call, so a wide budget there multiplies by
    # twelve. Adjudication runs only on the two to six dimensions actually in
    # dispute and asks one narrow question about each, so it can afford to read the
    # turns at close to full length -- which is what makes its verdict something
    # other than a confirmation of the blind pass's blind spot, and therefore
    # something the gate can count.
    adjudication_max_conversation_chars: int = 400000
    adjudication_max_comparison_chars: int = 600000
    adjudication_max_turn_chars: int = 100000

    # Check 450's prompts carry the task's own weighted rubric, and this bounds
    # it. A contributor who faults a model for not returning a Google Drive link
    # is quoting rubric criterion 811b1f24 (weight 4), not inventing a standard,
    # and a judge shown only the conversation reads the difference the wrong way
    # round: on task 6a7d3a48a9485b6a031cb312, a task human QC scored 5, all five
    # draws agreed on "inaccurate primary claim", which is a severe condition and
    # failed a clean task on its own.
    #
    # The budget is a guard rather than a routine trim. Across the 60-task
    # sheet22 run 15 tasks carried a rubric at all, of 18 to 40 criteria, and the
    # rendered block ran 4.8k characters at the median and 7.8k at the widest --
    # against a conversation budget of 160000 in the same prompt. Over the cap
    # the heaviest criteria are kept and the rest are dropped with a count, since
    # weight is the pipeline's own statement of which requirements matter.
    max_rubric_prompt_chars: int = 12000

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
    #
    # Check 450 reads it per model rather than per task, because its calls are
    # per model: one batched judgment covers all of that model's dimension
    # justifications, so a call that comes back empty takes a whole side of the
    # task with it and leaves a denominator that looks merely small. On the
    # 60-task sheet22 run three tasks audited 8 justifications of the 15 their
    # own dimension ratings carry -- on one of them (6a7d3a48a9485b6a031cb305)
    # model B's entire set was absent, the check reported clean, and a replay
    # with B present failed it at 3/15 with three severe items. A per-task ratio
    # would not have caught it: 8 of 15 is 53%, over this bar. A per-model one
    # does, because B's own coverage was zero.
    min_judged_ratio: float = 0.50

    def provenance(self) -> dict:
        d = asdict(self)
        d["policy_version"] = POLICY_VERSION
        return d


DEFAULT_POLICY = Policy()
