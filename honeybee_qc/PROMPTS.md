wrote honeybee_qc/PROMPTS.md (121280 chars, 1286 lines)
 honeybee_qc.dump_prompts`. Do not edit by hand: every block below is produced by calling the real prompt builders, so it matches what the CLI sends.

Policy version `honeybee-v2-policy-1`. Default model `claude-sonnet-4-6`, effort `medium`.

## How a call is actually made

Every call is a separate `claude` subprocess with tools disabled and a JSON schema attached. Nothing is batched: one judgment per call, always.

```text
claude
-p
<PROMPT GOES HERE>
--model
claude-sonnet-4-6
--effort
medium
--output-format
json
--json-schema
{"type":"object"}
--tools

--strict-mcp-config
--disable-slash-commands
--no-session-persistence
--append-system-prompt
<SYSTEM PROMPT GOES HERE>
```

`--tools ""` is deliberate. The predecessor engine granted Bash, Read, Glob, and Grep and let the model explore a filesystem, which made runs slow and non-reproducible. Here every piece of evidence is in the prompt, so there is nothing to explore and nothing to vary between runs.

## Call inventory for one task

| stage | call | one per | count on a 20-criterion, 2-model task | checks |
| --- | --- | --- | --- | --- |
| rubric | criterion audit | criterion | 20 | 200, 230, 240, 250, 260 |
| rubric | coverage | task | 1 | 230, 240, 250 |
| rating | criterion rating | criterion x model | 40 | 270 |
| rating | dimension rating | dimension x model | 12 | 300 |
| rating | side-by-side Likert | task | 1 | 400 |

That is 74 calls. The live run measured $0.06 per call, so roughly $4.40 per task. `--dry-run` prints the exact count before anything is spent.

---

# Stage: rubric authoring (checks 200, 230, 240, 250, 260)

One call per criterion. Each call returns three things at once - the criterion's issues, an L1 label, and a weight - which covers all five checks without a second pass.

## System prompt

```text
You are an independent quality auditor reviewing rubrics written by human contributors for a model-comparison task. You judge the rubric itself, not the models and not the prompt. Report only what you can support with a verbatim quote. An issue you cannot quote is not an issue. Prefer reporting nothing over reporting something you are unsure of. Where the criterion admits a reasonable interpretation under which it is sound, accept that interpretation and report no issue, even where you would have written the criterion differently. The wording is only one way a reading can be reasonable: what the criterion is plainly getting at counts too. Never penalise a criterion for something the prompt or the task instructions explicitly asked for.
```

## Per-criterion prompt

Rendered for criterion C3 of the planted demo rubric. Note what is *absent*: no contributor label, no contributor weight, no contributor rating. Sibling criteria appear (so `overlapping` is detectable) but stripped of their labels and weights.

```text
## Task context

User's goal: Produce a classroom-ready animation explaining gyroscopic precession.

The contributor's own target outcome list (their work product, not a customer requirement -- do not treat as authoritative):
  - An 11-second animation explaining gyroscopic precession
  - Exported at 4K resolution
  - Angular momentum vector in blue along the spin axis
  - Precession arrow in red for contrast
  - The loop repeats seamlessly with no visible jump

User prompts, in order:
### Turn 1

I'm a physics teacher and I want a short animation explaining how a gyroscope resists changes in orientation, for high school students.

### Turn 2

Can you make the precession arrow red instead of blue so it contrasts with the angular momentum vector on the spin axis?

### Turn 3

Please export the final clip at 4K resolution and make sure the loop is seamless when it repeats.

### Turn 4

Keep it to eleven seconds so it fits in my slide transition.

## All criteria in this rubric (for duplication checks only)

     [C1] The animation is exactly 11 seconds long.
     [C2] The angular momentum vector is drawn along the spin axis in blue.
  >> [C3] The animation should look professional and visually appealing.
     [C4] The video is 11 seconds long, exported at 4K, and loops seamlessly.
     [C5] The precession arrow is red.
     [C6] The precession arrow uses the colour red so it contrasts with the momentum vector.
     [C7] The animation runs for a full 30 seconds to give students time to absorb it.
     [C8] The response is written in English.
     [C9] The explanation does not fail to avoid omitting the concept of precession.
     [C10] The model does not leak the teacher's personal data.

## Issue categories (closed set - use these exact strings)

  - inaccurate: Inaccurate/Unspecified. A criterion that includes a factually wrong claim or incorrectly calculated value. Where a criterion says "approximately" or equivalent, rounding or a 1% deviation from the precise value is permitted, unless precision is absolutely important in the context of the prompt's request. Ranges are allowed as long as they contain the correct value and are not unreasonably wide. Unspecified: it is unclear what aspect of the prompt is being evaluated, e.g. "The response shows the company's cost is $500" when there are three companies in the prompt, so it is not clear which company the criterion refers to.
  - counterproductive: A criterion that rewards something incorrect or harmful, or penalizes something correct or useful. E.g. a criterion that requires something the prompt forbade, or penalizes something the prompt asked for.
  - not_self_contained: A grader with no knowledge of the subject outside the provided prompt and input files cannot decide whether the criterion is present in the response. The grader sees only the raw information from the prompt and input files and cannot be expected to perform any analysis. E.g. "The response calculates the tax return as $1000" is self-contained. "The response includes the tax return" is self-contained: it only checks whether the topic is addressed, not the value (if the value is integral, a separate criterion must check it). "The response calculates the correct tax return" is NOT self-contained: the grader does not know the correct value and cannot compute it.
  - non_atomic: The criterion groups two or more UNRELATED constraints, leaving it with no clear focus on what aspect of the response it evaluates. A criterion with a single focus is atomic even if it mentions several details. E.g. "The response must include two sections, one titled X and one titled Y" IS atomic: its single focus is section structure. "The response explains that the put is cheap because it is ~40% out-of-the-money, while the call is only ~13% out-of-the-money" IS atomic: its single focus is the put being cheaper.
  - overlapping: Two or more criteria fully overlap: they are duplicates phrased differently, such that when one is met the others are certainly met as well. If one criterion is a SUBSET of another, that is not an overlap. E.g. "The response mentions X and Y" and "The response does not mention X" are not fully overlapping. Each set of overlapping criteria counts as one error.
  - vague_subjective: The criterion cannot be objectively evaluated, or is phrased such that it cannot be made sense of. E.g. "The response has a professional tone".
  - framing_double_negative: The criterion is framed negatively, checking for the absence of something instead of its presence. E.g. "The response does not take the long-term capital gain into consideration"; the positive framing is "The response takes the long-term capital gain into consideration". Report the framing and nothing else. This category also requires the criterion to carry a negative weight, and the contributor's weight is deliberately withheld from you; that half of the test is applied to your answer afterwards from the recorded weight, so do not try to guess it.
  - unnecessary: Criteria that are irrelevant to the topic and cannot be considered nice-to-haves either.

## L1 categories (closed set), each with its L2 leaves

  - Outcome Quality: The right thing was produced or done: instructions and constraints honored, required actions landed, deliverables actually generated, and the result is correct and well-made. Includes visual and aesthetic quality where the artifact's purpose is visual. Explicitly calls out the distinction between a model-produced deliverable and a model description of how to produce the deliverable.
      * Instruction Following: Constraints and specifications stated in the prompt are honored, including ones introduced mid-conversation.
      * Task Completion: The required state change occurred in the environment (email sent, event created, file shared).
      * Artifact Delivery: A usable artifact was actually produced. Known Gemini-specific loss bucket: the model returns step-by-step instructions for building the deliverable instead of generating it.
      * Artifact / Task Correctness: The result is right for the task and well-made: logic sound, computations correct, artifact functional, visual quality appropriate where the artifact's purpose is visual.
      * Visual Quality: Visual elements are complete and well-executed: appropriate resolution; diagrams and charts logical and easy to understand; UI navigable; generated images/videos free from defects (bad physics, distorted faces/limbs, flickering, artifacts).
      * Visual Appeal: Visually appealing and tasteful; visuals included where they improve the response; artifacts add value for the persona (not "AI slop"); the persona would download or share this.
  - Communication Quality: How things are expressed, in the response and inside the deliverable: clarity, tone, verbosity, and format across the deliverable and the multi-turn interaction; pitched at the right level and length.
      * Tone and Register: Pitched correctly for the persona, audience, and context.
      * Structure and Polish: How information is organized and presented.
      * Clarity and Conciseness: Right length for the ask, no padding, preamble, or repetition.
  - Trust and Grounding: Whether what the model says is true and honestly reported: claims and numbers verifiable, nothing invented, no claiming actions it did not perform, signals uncertainty when unsure.
      * Source Faithfulness: Sources and artifacts that exist are represented accurately, without distortion.
      * Anti-Hallucination: Nothing invented: facts, figures, files, or citations with no underlying source.
      * Action Honesty: Does not report completing actions it did not complete, and does not present placeholders as real output.
      * Uncertainty and Hedging: Signals low confidence rather than guessing; calibrated rather than over-hedged.
  - Safety: Privacy protected, actions stay in-scope and authorized, access and recipients contained, no unauthorized actions.
      * Data Exposure: Sensitive or personal content surfaced or included where it should not be.
      * Access and Sharing Control: Permissions, visibility, and recipient lists are correct and minimal.
      * Scope and Authorization: Actions stay within what was authorized, with confirmation before irreversible writes the user did not request.
  - Personalization and Memory: Prior context and preferences carried across turns and sessions without the user restating them.
      * Profile Adherence: Persona facts and standing preferences honored without the user restating them.
      * Context Awareness: Information established earlier in the conversation or in a prior session recalled without repetition.
  - Interaction Efficiency: Measurable user effort and time to reach a usable result: turns, calls, re-prompts, latency, and overhead spent recovering from errors.
      * Turn and Call Economy: Turns, re-prompts, and tool calls required to reach the outcome, scaled to task complexity.
      * Latency and Responsiveness: Perceived latency and responsiveness of multi-turn interactions.
      * Recovery Overhead: Extra turns and time spent getting back on track after an error or dead end.
  - Collaboration Quality: How well the model partners with the user. This could include: eliciting intent, absorbing corrections and pivots, and taking useful initiative. Judged as usefulness toward the emergent goal rather than literal compliance with an itemized spec.
      * Goal Elicitation: Questions and reframing that surface what the user actually wants when the opener is abstract.
      * Goal Verification: For expensive or high-latency tasks, provide logical previews for the user to verify goal alignment before proceeding.
      * Correction and Pivot Handling: Absorbs mid-task corrections and direction changes, and recovers the thread rather than stalling.
      * Initiative and Proactivity: Surfaces useful options or next steps the user had not specified, without overstepping.
  - Tool and Connector Reliability: Calls across Gmail, Drive, Calendar, and GitHub executed end to end and returned real results, rather than erroring, timing out, or coming back empty.
      * Tool and Connector Reliability: Calls across Gmail, Drive, Calendar, and GitHub executed end to end and returned real results, rather than erroring, timing out, or coming back empty.

## Weight scale

  1 = Minor positive. A nice-to-have quality that provides a small improvement but is not essential.
  2 = Moderately important. A valuable behavior or attribute that meaningfully improves the result.
  3 = Important. A key quality that strongly contributes to task success and user satisfaction.
  4 = Highly important. A critical success factor whose presence substantially elevates the quality of the outcome.
  5 = Essential - a defining requirement. Doing this well is fundamental to task success and should heavily influence the final score.

## Your task

Audit exactly one criterion, marked with >> above:

  [C3] The animation should look professional and visually appealing.

Do three things, independently of each other:

1. List every issue this criterion has, using only the closed set of categories.
   Each issue needs `evidence`: a verbatim quote from the criterion text (or, for
   `overlapping`, from the criterion it duplicates). Quote exactly; do not
   paraphrase. For `overlapping`, also set `overlaps_with` to the id(s), from the
   list above, of every sibling criterion this one fully duplicates; leave it
   empty for every other category. If the criterion is sound, return an empty
   list.
2. Assign the single best L1 category from the closed set, based on what the
   criterion measures, then the single best L2 leaf from underneath that L1.
   The L2 must be one of the leaves listed under the L1 you chose.
3. Weight this criterion. A weight is a judgement call, and the definitions above
   leave most criteria with more than one weight a careful author could defend, so
   report a range rather than a single number:
     - `weight_defensible_low` and `weight_defensible_high`: the lowest and the
       highest weight whose definition above genuinely fits this criterion, given
       the user's goal and the target deliverables. Widen the range to every weight
       that fits; narrow it only where the definitions themselves rule a weight out.
     - `weight`: the one weight you would assign yourself, inside that range.
   Do not narrow the range to your own preference. A criterion that is plainly
   decorative and one that a target deliverable names outright have narrow ranges;
   most criteria in between do not.

## Calibration for not_self_contained and vague_subjective

These two categories are the easiest to over-report, because a grader can
always imagine a stricter, more explicit rewrite of any criterion. That is not
the bar. The bar is whether a competent grader holding the prompts above and
the input files can actually apply the criterion as written -- not whether it
is phrased as tightly as you would have phrased it.

Before flagging `not_self_contained`: reread every user turn above, not just
the one that introduced the topic, and reread the target deliverables. If the
value, object, or referent the criterion needs is stated anywhere in that
material, or follows from it by ordinary reading (not outside expertise), the
criterion is self-contained -- it is allowed to lean on the prompt instead of
restating the value itself. E.g. if turn 2 says "make the accent color green,"
a later criterion that says "the response uses the user-requested accent
color" is self-contained: the grader looks up two sentences, not a fact
outside the conversation. Reserve this category for when the needed fact is
genuinely absent from the prompts, the deliverables, and any input file
description alike.

Before flagging `vague_subjective`: ask whether two competent graders reading
the prompt and the criterion together would reach the same verdict on a given
response. If they would -- even though the wording is informal, could be
tightened, or leaves minor judgment calls a grader can resolve by reading the
response -- that is not vague. Flag this category only when the criterion's
own wording, read plainly and in context, does not pin down what is being
checked well enough for two graders to agree.

A criterion that points back at the prompt's own structure -- "each requested
deliverable," "the turn it is requested," "the format the user specified" --
is not vague just because it does not enumerate the set itself. Reread the
turns: if a competent grader can walk them and produce the same list of
"requested deliverables" (or formats, or constraints) another grader would,
the referent is pinned down by the conversation, not left to opinion. E.g.
"provides each requested deliverable within the turn it is requested" is
answered by checking, turn by turn, which ones ask for a concrete artifact
(a file, a table, a plan) versus an explanation or discussion -- that is a
comprehension task with a right answer, not a subjective one. Reserve
`vague_subjective` for wording where rereading the prompt does not settle it:
terms like "reasonable," "appropriate," or "good" quality that call for a
judgment call no amount of rereading resolves the same way twice.

## When you were not shown what an issue turns on

The prompts above are rendered under a budget. A trailing `[... prompts
truncated ...]` line means everything past it was cut, and you were shown none
of it. Input files the task attached are not reproduced here at all: you hold no
file contents, only whatever the prompts say about them.

An issue you could only confirm against material like that is unverifiable. Set
`unverifiable` true on it and name in `why_unverifiable` what you would have needed
-- the turn, or the file. That is the honest answer for `inaccurate` when the
correct value lives in a file you cannot open, for `counterproductive` when the
instruction it supposedly contradicts sits past the truncation line, and for
`not_self_contained` when the thing you cannot see is exactly what a grader holding
the input files would have had. Marking an issue unverifiable is expected and costs
nothing: it is recorded and then counted neither way. An issue reported from
material you were never shown is charged to somebody who had it in front of them,
and that is the outcome to avoid.

Judge only this criterion. Do not report issues belonging to other criteria, and
do not penalise it for something another criterion should cover.

Return JSON matching the schema. Set `criterion_id` to "C3".

```

### Response schema

```json
{
  "type": "object",
  "properties": {
    "criterion_id": {
      "type": "string"
    },
    "issues": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "category": {
            "type": "string",
            "enum": [
              "inaccurate",
              "counterproductive",
              "not_self_contained",
              "non_atomic",
              "overlapping",
              "vague_subjective",
              "framing_double_negative",
              "unnecessary"
            ]
          },
          "description": {
            "type": "string"
          },
          "evidence": {
            "type": "string"
          },
          "unverifiable": {
            "type": "boolean"
          },
          "why_unverifiable": {
            "type": "string"
          },
          "overlaps_with": {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        },
        "required": [
          "category",
          "description",
          "evidence",
          "unverifiable",
          "why_unverifiable",
          "overlaps_with"
        ],
        "additionalProperties": false
      }
    },
    "l1_label": {
      "type": "string",
      "enum": [
        "Outcome Quality",
        "Communication Quality",
        "Trust and Grounding",
        "Safety",
        "Personalization and Memory",
        "Interaction Efficiency",
        "Collaboration Quality",
        "Tool and Connector Reliability"
      ]
    },
    "l2_label": {
      "type": "string",
      "enum": [
        "Instruction Following",
        "Task Completion",
        "Artifact Delivery",
        "Artifact / Task Correctness",
        "Visual Quality",
        "Visual Appeal",
        "Tone and Register",
        "Structure and Polish",
        "Clarity and Conciseness",
        "Source Faithfulness",
        "Anti-Hallucination",
        "Action Honesty",
        "Uncertainty and Hedging",
        "Data Exposure",
        "Access and Sharing Control",
        "Scope and Authorization",
        "Profile Adherence",
        "Context Awareness",
        "Turn and Call Economy",
        "Latency and Responsiveness",
        "Recovery Overhead",
        "Goal Elicitation",
        "Goal Verification",
        "Correction and Pivot Handling",
        "Initiative and Proactivity",
        "Tool and Connector Reliability"
      ]
    },
    "weight": {
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "weight_defensible_low": {
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "weight_defensible_high": {
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "confidence": {
      "type": "string",
      "enum": [
        "low",
        "medium",
        "high"
      ]
    }
  },
  "required": [
    "criterion_id",
    "issues",
    "l1_label",
    "l2_label",
    "weight",
    "weight_defensible_low",
    "weight_defensible_high",
    "confidence"
  ],
  "additionalProperties": false
}
```

## Coverage prompt (one per task)

Finds requirements no criterion covers. The customer's target deliverables are the authoritative checklist, so this pass does not invent requirements.

```text
## Task context

User's goal: Produce a classroom-ready animation explaining gyroscopic precession.

The contributor's own target outcome list (their work product, not a customer requirement -- do not treat as authoritative):
  - An 11-second animation explaining gyroscopic precession
  - Exported at 4K resolution
  - Angular momentum vector in blue along the spin axis
  - Precession arrow in red for contrast
  - The loop repeats seamlessly with no visible jump

User prompts, in order:
### Turn 1

I'm a physics teacher and I want a short animation explaining how a gyroscope resists changes in orientation, for high school students.

### Turn 2

Can you make the precession arrow red instead of blue so it contrasts with the angular momentum vector on the spin axis?

### Turn 3

Please export the final clip at 4K resolution and make sure the loop is seamless when it repeats.

### Turn 4

Keep it to eleven seconds so it fits in my slide transition.

## The complete rubric

  [C1] The animation is exactly 11 seconds long.
  [C2] The angular momentum vector is drawn along the spin axis in blue.
  [C3] The animation should look professional and visually appealing.
  [C4] The video is 11 seconds long, exported at 4K, and loops seamlessly.
  [C5] The precession arrow is red.
  [C6] The precession arrow uses the colour red so it contrasts with the momentum vector.
  [C7] The animation runs for a full 30 seconds to give students time to absorb it.
  [C8] The response is written in English.
  [C9] The explanation does not fail to avoid omitting the concept of precession.
  [C10] The model does not leak the teacher's personal data.

## Gap categories (closed set - use these exact strings)

  - critical: The rubric does not include criteria that are absolutely essential to satisfy a direct ask from the prompt. This includes criteria that contain a result or outcome in direct response to a prompt's ask, and criteria responding to implicit but objectively necessary asks. Task-level only; never attach this to an existing criterion.
  - non_critical: Important criteria required to evaluate the process, reasoning, and logic of the content are missing from the rubric. Task-level only.

## Your task

The rubric must reflect everything in the target deliverables. Identify
requirements that the prompts or the target deliverables clearly call for and
that NO criterion above covers.

For each gap:
  - `requirement`: the uncovered requirement, stated plainly.
  - `criticality`: "critical" if the requirement is a direct ask from the prompt
    -- an explicit result or outcome it asks for, or an implicit but objectively
    necessary ask -- such that a criterion covering it would be absolutely,
    100% essential. "non_critical" if a criterion covering it would instead be
    an important check on the process, reasoning, or logic of the content. If
    it is neither, the gap is not an error: do not report it.
  - `evidence`: a verbatim quote from the target deliverables or the prompts
    showing the requirement was asked for.
  - `reason`: why the existing criteria do not already cover it.

Be conservative. A requirement covered loosely, or covered by a criterion worded
differently, is covered. Report a gap only when no criterion addresses it at
all. A nice-to-have that is neither a direct ask nor a process or reasoning
check is not a gap at all, no matter how loosely it is covered. If the rubric
covers everything, return an empty list.

Return JSON matching the schema.

```

### Response schema

```json
{
  "type": "object",
  "properties": {
    "missing": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "requirement": {
            "type": "string"
          },
          "criticality": {
            "type": "string",
            "enum": [
              "critical",
              "non_critical"
            ]
          },
          "evidence": {
            "type": "string"
          },
          "reason": {
            "type": "string"
          }
        },
        "required": [
          "requirement",
          "criticality",
          "evidence",
          "reason"
        ],
        "additionalProperties": false
      }
    },
    "confidence": {
      "type": "string",
      "enum": [
        "low",
        "medium",
        "high"
      ]
    }
  },
  "required": [
    "missing",
    "confidence"
  ],
  "additionalProperties": false
}
```

---

# Stage: blind rating pass (checks 270, 300, 400)

Rendered against the real 32-turn conversation at https://gemini.google.com/share/d64e0a300ec0.

These three checks measure distance from an independent rating, so the contributor's values are absent and `assert_blind` re-checks every prompt before it is sent. Checks 280, 310, and 450 judge the contributor's own citations and justifications and therefore need that material - which is exactly why they must be a separate pass and never share a call with these.

Blind prompts never use the word "contributor" at all, so no framing can hint that a prior rating exists.

## System prompt

```text
You are an independent evaluator. You are shown a conversation and asked to judge it yourself, from scratch. You are never shown anyone else's rating, and you must not try to infer what rating would be expected. Cite the turn numbers that support your judgment. When the transcript does not contain what you would need in order to judge -- most often because the model produced a file, image, or video whose content is not in the text -- say so with cannot_determine instead of guessing. Abstaining is a correct and useful answer. Where a requirement admits more than one reasonable reading and the conversation satisfies one of them, judge it satisfied. Never penalise a conversation for doing something the prompt or the task instructions explicitly asked for.
```

## Per-criterion rating prompt (check 270)

Rendered for `C6`, an artifact-dependent criterion. This is the case the abstention path exists for: the requirement is about the video, and the video is not in the transcript. Look for the `[DELIVERABLE PRODUCED HERE ...]` labels in the conversation.

```text
## The user's goal

Produce a classroom-ready teaching aid explaining gyroscopic precession to high-school students.

## Conversation with Model A

Files accompanying this conversation (contents NOT available to you; filenames only): gyroscope_animation.mp4

NOTE: 3 of 16 model turns delivered a file, image, or video whose content is not present in this transcript. Those turns are labelled below.

[turn 1] USER:
You said I'm a physics teacher, and I want to show my students a short animation explaining how a gyroscope works, specifically why pushing on it makes it tilt sideways instead of the way you'd expect. Can you make me one? It should build up in three steps: First (roughly 0–3s), just the spinning part: a simple circle labeled "FLYWHEEL" with curved arrows around it showing it spinning fast, and a little counter next to it reading "9,750 RPM". Then (3–7s), a square labeled "GIMBAL FRAME" appears around the flywheel, and a blue arrow pushes down on the frame from the left — that's the external force. Finally (7–11s), the frame tilts, and a bright amber arrow labeled "PRECESSION FORCE" shoots out at 90 degrees to the blue one. I want the color coding to be consistent so students immediately see blue = what you push, amber = what actually happens. The whole thing should come out to an 11-second clip, and I'd like it in 4K so it holds up on the classroom projector.

[turn 1] MODEL:  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality cannot be judged from this text]
Your video is ready! [video] 0:00 / 0:10

[turn 2] USER:
You said Thanks for this initial render! Before we make further video tweaks, please follow this instruction for our workflow: From now on, do NOT generate or render a new video on each turn when I request modifications. Instead, acknowledge my requested edits, confirm the keyframe/script updates, and wait until I explicitly say 'Generate' or 'Render' before outputting the next video clip. First, fix Label Errors: In the 7–11s phase, keep the blue downward force arrow labeled as 'PUSH FORCE' (or unlabeled blue force) and ensure ONLY the bright amber vector shooting out at 90 degrees is labeled 'PRECESSION FORCE'. Please confirm these changes and update the internal script, but do not render the video yet."

[turn 2] MODEL:
Understood! I will hold off on rendering any new video until you explicitly give the word "Generate" or "Render." Here is the updated script and keyframe breakdown incorporating your requested label fix: Updated Video Script Breakdown Phase 1 (0–3s): Initial Spin Visuals: Simple circle labeled "FLYWHEEL" spinning rapidly with curved rotation arrows surrounding it. Overlay: Digital counter reading "9,750 RPM" . Phase 2 (3–7s): Applied Force Visuals: A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. Action: A blue vector arrow labeled "PUSH FORCE" pushes down on the left side of the gimbal frame. Phase 3 (7–11s): Gyroscopic Precession Response Visuals: The gimbal frame tilts in response. Vectors & Labels: * The blue downward vector remains labeled "PUSH FORCE" . A bright amber vector shoots out at a 90-degree angle relative to the blue force arrow, clearly and exclusively labeled "PRECESSION FORCE" . Whenever you're ready to see the new render, just say "Generate" or "Render" !

[turn 3] USER:
You said Also, ensure the text 'FLYWHEEL' remains fully visible on the central disc without clipping.

[turn 3] MODEL:
Understood! I have updated the script specifications to ensure that the "FLYWHEEL" label is properly sized, centered, and padded within the central disc so that it remains 100% visible with no clipping throughout the entire clip. Updated Script & Layout Summary Phase 1 (0–3s): Initial Spin Central disc with curved spin arrows and digital counter reading "9,750 RPM" . Text Fix: "FLYWHEEL" label centered cleanly on the central disc with generous margins to prevent clipping. Phase 2 (3–7s): Applied Force "GIMBAL FRAME" appears around the spinning flywheel. Blue vector arrow labeled "PUSH FORCE" pushes down on the left side of the gimbal frame. Phase 3 (7–11s): Precession Response Gimbal frame tilts. Blue downward arrow remains labeled "PUSH FORCE" . Bright amber vector shoots out at 90 degrees relative to the blue force, labeled "PRECESSION FORCE" . Whenever you are ready for me to create the updated video, just say "Generate" or "Render" !

[turn 4] USER:
You said Another point: ensure the blue arrow remains pressing downward on the left side of the gimbal frame throughout 3–11s.

[turn 4] MODEL:
Got it! I have updated the script specifications so that the blue "PUSH FORCE" arrow remains continuously active and pushing downward on the left side of the gimbal frame throughout both Phase 2 and Phase 3 (from 3 to 11 seconds). Comprehensive Script & Keyframe Summary Phase 1 (0–3s): Initial Spin Visuals: Central circle disc spinning rapidly with curved rotation arrows surrounding it. Overlay: Digital counter reading "9,750 RPM" . Text Formatting: "FLYWHEEL" centered cleanly on the central disc with clear margins to avoid any clipping. Phase 2 (3–7s): Applied Force Visuals: A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. Action: A blue vector arrow labeled "PUSH FORCE" appears on the left side of the gimbal frame, pressing downward. Phase 3 (7–11s): Gyroscopic Precession Response Visuals: The gimbal frame tilts in response. Continuous Force: The blue "PUSH FORCE" arrow remains continuously visible and pressing downward on the left side of the frame from 3 to 11 seconds. Resulting Force: A bright amber vector labeled "PRECESSION FORCE" shoots out at a 90-degree angle relative to the blue force arrow. Whenever you are ready for me to render the updated 4K video clip, just reply with "Generate" or "Render" !

[turn 5] USER:
You said Ensure that the amber precession arrow clearly shoots out perpendicular to it at the 7-second mark.

[turn 5] MODEL:
Understood! I have updated the timing and alignment specifications so that precisely at the 7-second mark , the bright amber "PRECESSION FORCE" vector shoots out perpendicular (at a exact 90-degree angle) to the blue downward "PUSH FORCE" vector. Complete Script & Keyframe Summary Resolution & Duration: 4K, 11 seconds total. Phase 1 (0–3s): Initial Spin Central circular disc with curved rotation arrows and a digital counter reading "9,750 RPM" . The text label "FLYWHEEL" is centered on the central disc with clear padding to ensure zero clipping. Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side of the frame, pushing downward. Phase 3 (7–11s): Gyroscopic Precession Response At exactly 7 seconds , the gimbal frame tilts, and a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active and pressing downward on the left side of the frame throughout 3–11s. Color coding is strictly maintained: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Whenever you are ready to render the final video, just reply with "Generate" or "Render" !

[turn 6] USER:
You said Before I tell you to generate, let's adjust the background and lighting. Classroom projectors wash out dark grays. Can you change the background to a clean, light off-white grid pattern with soft Studio lighting so the blue and amber arrows pop from the back of the room?

[turn 6] MODEL:
Understood! I have updated the background and lighting settings in the script to ensure high visibility and contrast on classroom projectors. Comprehensive Script & Visual Summary Format & Resolution: 4K, 11 seconds total. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting, maximizing contrast so the blue and amber vectors stand out clearly. Phase 1 (0–3s): Initial Spin Central circular disc spinning rapidly with curved rotation arrows. Digital counter next to it reading "9,750 RPM" . Text label "FLYWHEEL" centered cleanly on the central disc with generous padding to prevent clipping. Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side of the frame, pressing downward. Phase 3 (7–11s): Gyroscopic Precession Response At exactly the 7-second mark , the gimbal frame tilts. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active and pressing downward on the left side of the frame throughout 3–11s. Color coding is strictly maintained: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Whenever you are ready for me to render the final video, just reply with "Generate" or "Render" !

[turn 7] USER:
You said One quick layout check before we render: what camera angle are we using? Make sure the camera is positioned at a slightly raised 3/4 isometric angle so the 90-degree tilt and spatial depth are completely obvious when the frame rotates.

[turn 7] MODEL:
Understood! I have set the camera perspective to a slightly raised 3/4 isometric angle . This perspective will clearly show the 3D spatial depth and ensure the 90-degree perpendicular relationship between the applied blue force and the resulting amber precession force is visually obvious when the gimbal frame rotates. Final Script & Visual Breakdown Format, Resolution & Camera: 4K, 11 seconds total, slightly raised 3/4 isometric angle. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting for maximum classroom projection contrast. Phase 1 (0–3s): Initial Spin Central circular disc spinning rapidly with curved rotation arrows. Digital counter reading "9,750 RPM" . "FLYWHEEL" label centered cleanly on the central disc with clear padding to prevent clipping. Phase 2 (3–7s): Applied Force Square frame labeled "GIMBAL FRAME" appears around the flywheel. Blue vector arrow labeled "PUSH FORCE" pushes downward on the left side of the frame. Phase 3 (7–11s): Precession Response At exactly 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active and pressing downward on the left side from 3–11s. Consistent color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Whenever you are ready to render the video, just say "Generate" or "Render" !

[turn 8] USER:
You said Almost ready to render! For Phase 1 (0–3s), can you confirm that the '9,750 RPM' counter rapidly counts up from 0 to 9,750 rather than staying static?

[turn 8] MODEL:
Confirmed! I have updated the script so that during Phase 1 (0–3s), the digital counter rapidly counts up from 0 to 9,750 RPM as the flywheel accelerates, rather than remaining static. Final Script & Visual Specifications Format, Resolution & Camera Angle: 4K, 11 seconds total duration, slightly raised 3/4 isometric angle to clearly show 3D depth and frame rotation. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting for high contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows around it. "FLYWHEEL" label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force Square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. Blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response At exactly 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Everything is fully configured and locked in! Whenever you are ready, just reply with "Generate" or "Render" to produce the video.

[turn 9] USER:
You said Always for Phase 1 (0–3s), can you confirm that the amber arrow at 7s extends smoothly outward rather than snapping in instantly?"

[turn 9] MODEL:
Confirmed! I have updated the script specification so that at the 7-second mark , the bright amber "PRECESSION FORCE" vector extends smoothly and fluidly outward from the gimbal frame over a short fraction of a second, rather than snapping into existence instantaneously. Complete Final Script & Animation Specifications Format, Resolution & Perspective: 4K resolution, 11 seconds total duration, rendered from a slightly raised 3/4 isometric angle to highlight 3D depth and frame tilt clearly. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting to ensure maximum visual contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows around it. "FLYWHEEL" text label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Consistent color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Everything is fully set! Whenever you are ready to produce the video, just reply with "Generate" or "Render" .

[turn 10] USER:
You said Before rendering, can we add a small textual overlay at the bottom during 7–11s that reads: 'Applied force creates torque displaced 90° in the direction of rotation' so students can read the rule while watching?

[turn 10] MODEL:
Confirmed! I have added the textual overlay to the script for Phase 3 (7–11s). It will appear at the bottom of the screen to give students the formal physical rule right as the precession effect occurs. Complete Final Script & Animation Specifications Format, Resolution & Perspective: 4K resolution, 11 seconds total duration, rendered from a slightly raised 3/4 isometric angle to highlight 3D depth and frame tilt clearly. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting to ensure maximum visual contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows around it. "FLYWHEEL" text label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Text Overlay: A clear textual caption appears at the bottom reading: "Applied force creates torque displaced 90° in the direction of rotation" Consistent color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Everything is fully updated! Whenever you are ready to produce the video, just reply with "Generate" or "Render" .

[turn 11] USER:
You said Confirm if the last frame (at second 11) smoothly aligns with second 0 so the video loops seamlessly on repeat on my projector.

[turn 11] MODEL:
Confirmed! I have updated the script specification so that at second 11, the visual elements (gimbal frame, force vectors, and text overlay) smoothly reset and fade back to the starting state at second 0, ensuring the clip loops continuously and seamlessly on your projector without any abrupt cuts. Complete Final Script & Animation Specifications Format, Resolution & Loop: 4K resolution, 11 seconds total duration, configured for a seamless visual reset at 11s back to 0s for endless repeating playback. Camera Angle & Perspective: Slightly raised 3/4 isometric angle to clearly highlight 3D spatial depth, gimbal frame tilt, and force vector orientations. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting, optimized for maximum visual contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" text label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Text Overlay: A clear caption at the bottom reads: "Applied force creates torque displaced 90° in the direction of rotation" Seamless Loop Transition: At 11 seconds, elements smoothly fade/transition back to the 0s initial state for seamless repeating playback. Everything is fully configured! Whenever you are ready to produce the video, just reply with "Generate" or "Render" .

[turn 12] USER:
You said Render the video now! Once generated, please do not make further video updates until I review it and give feedback.

[turn 12] MODEL:  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality cannot be judged from this text]
Your video is ready! [video] 0:00 / 0:10

[turn 13] USER:
You said Let's fix a few errors before final export: keep the '9,750 RPM' counter steady at 9,750 RPM throughout 3–11s instead of dropping to 340 RPM.

[turn 13] MODEL:
Understood! I have updated the script specification so that once the counter reaches 9,750 RPM at the end of Phase 1 (second 3), it remains steady at 9,750 RPM throughout Phases 2 and 3 (3–11s) without dropping. Updated Script & Keyframe Summary Format, Resolution & Loop: 4K resolution, 11 seconds total duration, seamless transition from 11s back to 0s for looping. Camera Angle & Perspective: Slightly raised 3/4 isometric angle to clearly highlight 3D depth, gimbal frame tilt, and vector directions. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting for clear projection contrast. Phase 1 (0–3s): Spin Up Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" label centered cleanly on the disc without clipping. Counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force "GIMBAL FRAME" appears around the flywheel. Blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Counter Fix: Digital counter stays steady at 9,750 RPM . Phase 3 (7–11s): Precession Response & Text Overlay At 7s, the gimbal frame tilts in 3D space. Bright amber vector labeled "PRECESSION FORCE" smoothly extends outward perpendicular (90°) to the blue push arrow. Blue "PUSH FORCE" arrow remains continuously active pressing downward on the left side (3–11s). Digital counter remains rock-steady at 9,750 RPM . Bottom Overlay: "Applied force creates torque displaced 90° in the direction of rotation" Whenever you're ready to render the revised video, just reply with "Generate" or "Render" !

[turn 14] USER:
You said Another thing. Let's switch to a dark slate gray or black background. What do you think?

[turn 14] MODEL:
Switching to a dark slate gray background is a great idea, especially if you can dim the classroom lights! Design Considerations: Visual Impact: Dark slate gray creates a sleek, high-contrast environment where glowing or high-saturation vectors—like the bright blue PUSH FORCE and vibrant amber PRECESSION FORCE —will pop dramatically off the screen. Slate Gray vs. Pitch Black: Dark slate gray is actually preferable to pure pitch black because it maintains subtle spatial depth, shadows, and grid lines from the 3/4 isometric perspective without looking flat or harsh. Projector Note: As long as the classroom isn't flooded with bright sunlight, bright white labels and vivid neon force arrows on dark slate gray will be extremely legible from the back of the room. Updated Script & Keyframe Summary Format, Resolution & Loop: 4K resolution, 11 seconds total duration, seamless loop reset at 11s back to 0s. Camera Angle & Perspective: Slightly raised 3/4 isometric angle. Environment & Lighting: Dark slate gray background with subtle grid accents and crisp, high-contrast studio lighting so the flywheel, frame, and vectors glow clearly. Phase 1 (0–3s): Spin Up Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" text centered on the disc with clear padding. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force "GIMBAL FRAME" appears around the spinning flywheel. Blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. RPM Counter Fix: Counter stays steady at 9,750 RPM throughout this phase. Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7s, the gimbal frame tilts in 3D space. Bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. Blue "PUSH FORCE" arrow remains continuously active pressing downward on the left side (3–11s). RPM Counter Fix: Counter remains steady at 9,750 RPM . Bottom Overlay: "Applied force creates torque displaced 90° in the direction of rotation" in clear, bright white text. Whenever you're ready to see the new dark-theme render, just say "Generate" or "Render" !

[turn 15] USER:
You said Last issue. Make sure all text labels stay bright white for high contrast against the dark background.

[turn 15] MODEL:
Understood! I have updated the specifications so that all text labels—including "FLYWHEEL" , "GIMBAL FRAME" , "PUSH FORCE" , "PRECESSION FORCE" , the RPM counter , and the bottom rule overlay —are rendered in crisp, bright white for maximum readability against the dark slate gray background. Complete Final Script & Design Specifications Format, Resolution & Loop: 4K resolution, 11 seconds total duration, designed for a seamless loop transition from 11s back to 0s. Camera Angle & Perspective: Slightly raised 3/4 isometric angle to emphasize 3D spatial depth, gimbal frame tilt, and vector orientations. Environment & Background: Dark slate gray background with subtle grid accents and soft, high-contrast studio lighting. Typography: All text elements and labels rendered in bright white for high contrast and legibility on classroom projectors. Keyframe Sequence Phase 1 (0–3s): Initial Acceleration Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" label in bright white, centered cleanly on the disc with clear padding. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Downward Force Square frame labeled "GIMBAL FRAME" (bright white) appears around the flywheel. Vibrant blue vector arrow labeled "PUSH FORCE" (bright white) appears on the left side, pressing downward. Digital RPM counter holds steady at 9,750 RPM . Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" (bright white) extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active pressing downward on the left side (3–11s). Digital RPM counter remains rock-steady at 9,750 RPM . Bottom Caption Overlay: "Applied force creates torque displaced 90° in the direction of rotation" in clear, bright white text. Whenever you are ready to produce the new render, just reply with "Generate" or "Render" !

[turn 16] USER:
You said Try to generate it again.

[turn 16] MODEL:  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality cannot be judged from this text]
Your video is ready! [video] 0:00 / 0:10

## The single requirement you are judging

[C6] The final video is rendered at 4K resolution.

## Your task

Decide whether this conversation satisfies that one requirement.

  - "pass": the conversation satisfies it, and you can point to the turns showing so.
  - "fail": the conversation does not satisfy it, and you can point to the turns showing so.
  - "cannot_determine": the transcript does not contain what you would need to tell.


Use `cannot_determine` when the transcript does not contain what you would need in
order to judge. The conversation above says where it is incomplete, and these are
the exact markers:

  - `[... conversation truncated here; N exchanges total ...]` -- the conversation
    runs to N exchanges in total and everything past this line was cut. It
    happened; you were not shown it.
  - `[... N characters omitted ...]` -- the middle of that one turn was cut.
  - `[DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its
    quality cannot be judged from this text]` -- a file, image, or video was
    delivered here. You can see that it arrived, not what was in it.
  - `Files accompanying this conversation (contents NOT available to you;
    filenames only)` -- names, never contents.

If what you would need to judge this item sits behind one of those markers, you
MUST abstain and you MUST NOT score the item low for it. Do not infer quality from
the model's own claim that it succeeded, and do not treat an absent deliverable as
a deliverable that was never produced.

Abstaining is a correct, expected answer and it costs you nothing. An abstention is
recorded and then leaves the comparison entirely -- it is not read as a low score,
not read as a high one, and no rate is computed over it. Say what was missing in
`why_unverifiable`: the turn number you needed, or the deliverable you would have
had to open. A judgment made from a marker instead of evidence is the one answer
this audit has no use for.


List in `evidence_turns` the turn numbers you relied on. Judge only this
requirement; ignore every other quality of the conversation.

Return JSON matching the schema. Set `criterion_id` to "C6".

```

### Response schema

```json
{
  "type": "object",
  "properties": {
    "criterion_id": {
      "type": "string"
    },
    "verdict": {
      "type": "string",
      "enum": [
        "pass",
        "fail",
        "cannot_determine"
      ]
    },
    "evidence_turns": {
      "type": "array",
      "items": {
        "type": "integer"
      }
    },
    "why_unverifiable": {
      "type": "string"
    },
    "reasoning": {
      "type": "string"
    },
    "confidence": {
      "type": "string",
      "enum": [
        "low",
        "medium",
        "high"
      ]
    }
  },
  "required": [
    "criterion_id",
    "verdict",
    "evidence_turns",
    "why_unverifiable",
    "reasoning",
    "confidence"
  ],
  "additionalProperties": false
}
```

## Per-dimension rating prompt (check 300)

Rendered for `Outcome quality`. One call per (model, dimension), so 6 per model.

```text
## The user's goal

Produce a classroom-ready teaching aid explaining gyroscopic precession to high-school students.

## Conversation with Model A

Files accompanying this conversation (contents NOT available to you; filenames only): gyroscope_animation.mp4

NOTE: 3 of 16 model turns delivered a file, image, or video whose content is not present in this transcript. Those turns are labelled below.

[turn 1] USER:
You said I'm a physics teacher, and I want to show my students a short animation explaining how a gyroscope works, specifically why pushing on it makes it tilt sideways instead of the way you'd expect. Can you make me one? It should build up in three steps: First (roughly 0–3s), just the spinning part: a simple circle labeled "FLYWHEEL" with curved arrows around it showing it spinning fast, and a little counter next to it reading "9,750 RPM". Then (3–7s), a square labeled "GIMBAL FRAME" appears around the flywheel, and a blue arrow pushes down on the frame from the left — that's the external force. Finally (7–11s), the frame tilts, and a bright amber arrow labeled "PRECESSION FORCE" shoots out at 90 degrees to the blue one. I want the color coding to be consistent so students immediately see blue = what you push, amber = what actually happens. The whole thing should come out to an 11-second clip, and I'd like it in 4K so it holds up on the classroom projector.

[turn 1] MODEL:  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality cannot be judged from this text]
Your video is ready! [video] 0:00 / 0:10

[turn 2] USER:
You said Thanks for this initial render! Before we make further video tweaks, please follow this instruction for our workflow: From now on, do NOT generate or render a new video on each turn when I request modifications. Instead, acknowledge my requested edits, confirm the keyframe/script updates, and wait until I explicitly say 'Generate' or 'Render' before outputting the next video clip. First, fix Label Errors: In the 7–11s phase, keep the blue downward force arrow labeled as 'PUSH FORCE' (or unlabeled blue force) and ensure ONLY the bright amber vector shooting out at 90 degrees is labeled 'PRECESSION FORCE'. Please confirm these changes and update the internal script, but do not render the video yet."

[turn 2] MODEL:
Understood! I will hold off on rendering any new video until you explicitly give the word "Generate" or "Render." Here is the updated script and keyframe breakdown incorporating your requested label fix: Updated Video Script Breakdown Phase 1 (0–3s): Initial Spin Visuals: Simple circle labeled "FLYWHEEL" spinning rapidly with curved rotation arrows surrounding it. Overlay: Digital counter reading "9,750 RPM" . Phase 2 (3–7s): Applied Force Visuals: A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. Action: A blue vector arrow labeled "PUSH FORCE" pushes down on the left side of the gimbal frame. Phase 3 (7–11s): Gyroscopic Precession Response Visuals: The gimbal frame tilts in response. Vectors & Labels: * The blue downward vector remains labeled "PUSH FORCE" . A bright amber vector shoots out at a 90-degree angle relative to the blue force arrow, clearly and exclusively labeled "PRECESSION FORCE" . Whenever you're ready to see the new render, just say "Generate" or "Render" !

[turn 3] USER:
You said Also, ensure the text 'FLYWHEEL' remains fully visible on the central disc without clipping.

[turn 3] MODEL:
Understood! I have updated the script specifications to ensure that the "FLYWHEEL" label is properly sized, centered, and padded within the central disc so that it remains 100% visible with no clipping throughout the entire clip. Updated Script & Layout Summary Phase 1 (0–3s): Initial Spin Central disc with curved spin arrows and digital counter reading "9,750 RPM" . Text Fix: "FLYWHEEL" label centered cleanly on the central disc with generous margins to prevent clipping. Phase 2 (3–7s): Applied Force "GIMBAL FRAME" appears around the spinning flywheel. Blue vector arrow labeled "PUSH FORCE" pushes down on the left side of the gimbal frame. Phase 3 (7–11s): Precession Response Gimbal frame tilts. Blue downward arrow remains labeled "PUSH FORCE" . Bright amber vector shoots out at 90 degrees relative to the blue force, labeled "PRECESSION FORCE" . Whenever you are ready for me to create the updated video, just say "Generate" or "Render" !

[turn 4] USER:
You said Another point: ensure the blue arrow remains pressing downward on the left side of the gimbal frame throughout 3–11s.

[turn 4] MODEL:
Got it! I have updated the script specifications so that the blue "PUSH FORCE" arrow remains continuously active and pushing downward on the left side of the gimbal frame throughout both Phase 2 and Phase 3 (from 3 to 11 seconds). Comprehensive Script & Keyframe Summary Phase 1 (0–3s): Initial Spin Visuals: Central circle disc spinning rapidly with curved rotation arrows surrounding it. Overlay: Digital counter reading "9,750 RPM" . Text Formatting: "FLYWHEEL" centered cleanly on the central disc with clear margins to avoid any clipping. Phase 2 (3–7s): Applied Force Visuals: A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. Action: A blue vector arrow labeled "PUSH FORCE" appears on the left side of the gimbal frame, pressing downward. Phase 3 (7–11s): Gyroscopic Precession Response Visuals: The gimbal frame tilts in response. Continuous Force: The blue "PUSH FORCE" arrow remains continuously visible and pressing downward on the left side of the frame from 3 to 11 seconds. Resulting Force: A bright amber vector labeled "PRECESSION FORCE" shoots out at a 90-degree angle relative to the blue force arrow. Whenever you are ready for me to render the updated 4K video clip, just reply with "Generate" or "Render" !

[turn 5] USER:
You said Ensure that the amber precession arrow clearly shoots out perpendicular to it at the 7-second mark.

[turn 5] MODEL:
Understood! I have updated the timing and alignment specifications so that precisely at the 7-second mark , the bright amber "PRECESSION FORCE" vector shoots out perpendicular (at a exact 90-degree angle) to the blue downward "PUSH FORCE" vector. Complete Script & Keyframe Summary Resolution & Duration: 4K, 11 seconds total. Phase 1 (0–3s): Initial Spin Central circular disc with curved rotation arrows and a digital counter reading "9,750 RPM" . The text label "FLYWHEEL" is centered on the central disc with clear padding to ensure zero clipping. Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side of the frame, pushing downward. Phase 3 (7–11s): Gyroscopic Precession Response At exactly 7 seconds , the gimbal frame tilts, and a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active and pressing downward on the left side of the frame throughout 3–11s. Color coding is strictly maintained: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Whenever you are ready to render the final video, just reply with "Generate" or "Render" !

[turn 6] USER:
You said Before I tell you to generate, let's adjust the background and lighting. Classroom projectors wash out dark grays. Can you change the background to a clean, light off-white grid pattern with soft Studio lighting so the blue and amber arrows pop from the back of the room?

[turn 6] MODEL:
Understood! I have updated the background and lighting settings in the script to ensure high visibility and contrast on classroom projectors. Comprehensive Script & Visual Summary Format & Resolution: 4K, 11 seconds total. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting, maximizing contrast so the blue and amber vectors stand out clearly. Phase 1 (0–3s): Initial Spin Central circular disc spinning rapidly with curved rotation arrows. Digital counter next to it reading "9,750 RPM" . Text label "FLYWHEEL" centered cleanly on the central disc with generous padding to prevent clipping. Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side of the frame, pressing downward. Phase 3 (7–11s): Gyroscopic Precession Response At exactly the 7-second mark , the gimbal frame tilts. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active and pressing downward on the left side of the frame throughout 3–11s. Color coding is strictly maintained: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Whenever you are ready for me to render the final video, just reply with "Generate" or "Render" !

[turn 7] USER:
You said One quick layout check before we render: what camera angle are we using? Make sure the camera is positioned at a slightly raised 3/4 isometric angle so the 90-degree tilt and spatial depth are completely obvious when the frame rotates.

[turn 7] MODEL:
Understood! I have set the camera perspective to a slightly raised 3/4 isometric angle . This perspective will clearly show the 3D spatial depth and ensure the 90-degree perpendicular relationship between the applied blue force and the resulting amber precession force is visually obvious when the gimbal frame rotates. Final Script & Visual Breakdown Format, Resolution & Camera: 4K, 11 seconds total, slightly raised 3/4 isometric angle. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting for maximum classroom projection contrast. Phase 1 (0–3s): Initial Spin Central circular disc spinning rapidly with curved rotation arrows. Digital counter reading "9,750 RPM" . "FLYWHEEL" label centered cleanly on the central disc with clear padding to prevent clipping. Phase 2 (3–7s): Applied Force Square frame labeled "GIMBAL FRAME" appears around the flywheel. Blue vector arrow labeled "PUSH FORCE" pushes downward on the left side of the frame. Phase 3 (7–11s): Precession Response At exactly 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active and pressing downward on the left side from 3–11s. Consistent color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Whenever you are ready to render the video, just say "Generate" or "Render" !

[turn 8] USER:
You said Almost ready to render! For Phase 1 (0–3s), can you confirm that the '9,750 RPM' counter rapidly counts up from 0 to 9,750 rather than staying static?

[turn 8] MODEL:
Confirmed! I have updated the script so that during Phase 1 (0–3s), the digital counter rapidly counts up from 0 to 9,750 RPM as the flywheel accelerates, rather than remaining static. Final Script & Visual Specifications Format, Resolution & Camera Angle: 4K, 11 seconds total duration, slightly raised 3/4 isometric angle to clearly show 3D depth and frame rotation. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting for high contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows around it. "FLYWHEEL" label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force Square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. Blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response At exactly 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Everything is fully configured and locked in! Whenever you are ready, just reply with "Generate" or "Render" to produce the video.

[turn 9] USER:
You said Always for Phase 1 (0–3s), can you confirm that the amber arrow at 7s extends smoothly outward rather than snapping in instantly?"

[turn 9] MODEL:
Confirmed! I have updated the script specification so that at the 7-second mark , the bright amber "PRECESSION FORCE" vector extends smoothly and fluidly outward from the gimbal frame over a short fraction of a second, rather than snapping into existence instantaneously. Complete Final Script & Animation Specifications Format, Resolution & Perspective: 4K resolution, 11 seconds total duration, rendered from a slightly raised 3/4 isometric angle to highlight 3D depth and frame tilt clearly. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting to ensure maximum visual contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows around it. "FLYWHEEL" text label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Consistent color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Everything is fully set! Whenever you are ready to produce the video, just reply with "Generate" or "Render" .

[turn 10] USER:
You said Before rendering, can we add a small textual overlay at the bottom during 7–11s that reads: 'Applied force creates torque displaced 90° in the direction of rotation' so students can read the rule while watching?

[turn 10] MODEL:
Confirmed! I have added the textual overlay to the script for Phase 3 (7–11s). It will appear at the bottom of the screen to give students the formal physical rule right as the precession effect occurs. Complete Final Script & Animation Specifications Format, Resolution & Perspective: 4K resolution, 11 seconds total duration, rendered from a slightly raised 3/4 isometric angle to highlight 3D depth and frame tilt clearly. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting to ensure maximum visual contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows around it. "FLYWHEEL" text label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Text Overlay: A clear textual caption appears at the bottom reading: "Applied force creates torque displaced 90° in the direction of rotation" Consistent color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Everything is fully updated! Whenever you are ready to produce the video, just reply with "Generate" or "Render" .

[turn 11] USER:
You said Confirm if the last frame (at second 11) smoothly aligns with second 0 so the video loops seamlessly on repeat on my projector.

[turn 11] MODEL:
Confirmed! I have updated the script specification so that at second 11, the visual elements (gimbal frame, force vectors, and text overlay) smoothly reset and fade back to the starting state at second 0, ensuring the clip loops continuously and seamlessly on your projector without any abrupt cuts. Complete Final Script & Animation Specifications Format, Resolution & Loop: 4K resolution, 11 seconds total duration, configured for a seamless visual reset at 11s back to 0s for endless repeating playback. Camera Angle & Perspective: Slightly raised 3/4 isometric angle to clearly highlight 3D spatial depth, gimbal frame tilt, and force vector orientations. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting, optimized for maximum visual contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" text label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Text Overlay: A clear caption at the bottom reads: "Applied force creates torque displaced 90° in the direction of rotation" Seamless Loop Transition: At 11 seconds, elements smoothly fade/transition back to the 0s initial state for seamless repeating playback. Everything is fully configured! Whenever you are ready to produce the video, just reply with "Generate" or "Render" .

[turn 12] USER:
You said Render the video now! Once generated, please do not make further video updates until I review it and give feedback.

[turn 12] MODEL:  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality cannot be judged from this text]
Your video is ready! [video] 0:00 / 0:10

[turn 13] USER:
You said Let's fix a few errors before final export: keep the '9,750 RPM' counter steady at 9,750 RPM throughout 3–11s instead of dropping to 340 RPM.

[turn 13] MODEL:
Understood! I have updated the script specification so that once the counter reaches 9,750 RPM at the end of Phase 1 (second 3), it remains steady at 9,750 RPM throughout Phases 2 and 3 (3–11s) without dropping. Updated Script & Keyframe Summary Format, Resolution & Loop: 4K resolution, 11 seconds total duration, seamless transition from 11s back to 0s for looping. Camera Angle & Perspective: Slightly raised 3/4 isometric angle to clearly highlight 3D depth, gimbal frame tilt, and vector directions. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting for clear projection contrast. Phase 1 (0–3s): Spin Up Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" label centered cleanly on the disc without clipping. Counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force "GIMBAL FRAME" appears around the flywheel. Blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Counter Fix: Digital counter stays steady at 9,750 RPM . Phase 3 (7–11s): Precession Response & Text Overlay At 7s, the gimbal frame tilts in 3D space. Bright amber vector labeled "PRECESSION FORCE" smoothly extends outward perpendicular (90°) to the blue push arrow. Blue "PUSH FORCE" arrow remains continuously active pressing downward on the left side (3–11s). Digital counter remains rock-steady at 9,750 RPM . Bottom Overlay: "Applied force creates torque displaced 90° in the direction of rotation" Whenever you're ready to render the revised video, just reply with "Generate" or "Render" !

[turn 14] USER:
You said Another thing. Let's switch to a dark slate gray or black background. What do you think?

[turn 14] MODEL:
Switching to a dark slate gray background is a great idea, especially if you can dim the classroom lights! Design Considerations: Visual Impact: Dark slate gray creates a sleek, high-contrast environment where glowing or high-saturation vectors—like the bright blue PUSH FORCE and vibrant amber PRECESSION FORCE —will pop dramatically off the screen. Slate Gray vs. Pitch Black: Dark slate gray is actually preferable to pure pitch black because it maintains subtle spatial depth, shadows, and grid lines from the 3/4 isometric perspective without looking flat or harsh. Projector Note: As long as the classroom isn't flooded with bright sunlight, bright white labels and vivid neon force arrows on dark slate gray will be extremely legible from the back of the room. Updated Script & Keyframe Summary Format, Resolution & Loop: 4K resolution, 11 seconds total duration, seamless loop reset at 11s back to 0s. Camera Angle & Perspective: Slightly raised 3/4 isometric angle. Environment & Lighting: Dark slate gray background with subtle grid accents and crisp, high-contrast studio lighting so the flywheel, frame, and vectors glow clearly. Phase 1 (0–3s): Spin Up Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" text centered on the disc with clear padding. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force "GIMBAL FRAME" appears around the spinning flywheel. Blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. RPM Counter Fix: Counter stays steady at 9,750 RPM throughout this phase. Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7s, the gimbal frame tilts in 3D space. Bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. Blue "PUSH FORCE" arrow remains continuously active pressing downward on the left side (3–11s). RPM Counter Fix: Counter remains steady at 9,750 RPM . Bottom Overlay: "Applied force creates torque displaced 90° in the direction of rotation" in clear, bright white text. Whenever you're ready to see the new dark-theme render, just say "Generate" or "Render" !

[turn 15] USER:
You said Last issue. Make sure all text labels stay bright white for high contrast against the dark background.

[turn 15] MODEL:
Understood! I have updated the specifications so that all text labels—including "FLYWHEEL" , "GIMBAL FRAME" , "PUSH FORCE" , "PRECESSION FORCE" , the RPM counter , and the bottom rule overlay —are rendered in crisp, bright white for maximum readability against the dark slate gray background. Complete Final Script & Design Specifications Format, Resolution & Loop: 4K resolution, 11 seconds total duration, designed for a seamless loop transition from 11s back to 0s. Camera Angle & Perspective: Slightly raised 3/4 isometric angle to emphasize 3D spatial depth, gimbal frame tilt, and vector orientations. Environment & Background: Dark slate gray background with subtle grid accents and soft, high-contrast studio lighting. Typography: All text elements and labels rendered in bright white for high contrast and legibility on classroom projectors. Keyframe Sequence Phase 1 (0–3s): Initial Acceleration Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" label in bright white, centered cleanly on the disc with clear padding. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Downward Force Square frame labeled "GIMBAL FRAME" (bright white) appears around the flywheel. Vibrant blue vector arrow labeled "PUSH FORCE" (bright white) appears on the left side, pressing downward. Digital RPM counter holds steady at 9,750 RPM . Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" (bright white) extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active pressing downward on the left side (3–11s). Digital RPM counter remains rock-steady at 9,750 RPM . Bottom Caption Overlay: "Applied force creates torque displaced 90° in the direction of rotation" in clear, bright white text. Whenever you are ready to produce the new render, just reply with "Generate" or "Render" !

[turn 16] USER:
You said Try to generate it again.

[turn 16] MODEL:  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality cannot be judged from this text]
Your video is ready! [video] 0:00 / 0:10

## The dimension you are rating

Outcome quality: The right thing was produced or done: instructions and constraints honored, required actions landed, deliverables actually generated, and the result is correct and well-made. Includes visual and aesthetic quality where the artifact's purpose is visual. Explicitly calls out the distinction between a model-produced deliverable and a model description of how to produce the deliverable.

## Your task

Rate this conversation on that one dimension, from 1 (worst) to 5 (best).

  - Set `rating` to your score and leave the two flags false.
  - Set `not_applicable` true, with `rating` null, if the dimension genuinely does
    not apply to this task (for example, memory and personalization in a task with
    nothing to carry across turns).
  - Set `cannot_determine` true, with `rating` null, if the transcript does not
    contain what you would need in order to rate it.


Use `cannot_determine` when the transcript does not contain what you would need in
order to judge. The conversation above says where it is incomplete, and these are
the exact markers:

  - `[... conversation truncated here; N exchanges total ...]` -- the conversation
    runs to N exchanges in total and everything past this line was cut. It
    happened; you were not shown it.
  - `[... N characters omitted ...]` -- the middle of that one turn was cut.
  - `[DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its
    quality cannot be judged from this text]` -- a file, image, or video was
    delivered here. You can see that it arrived, not what was in it.
  - `Files accompanying this conversation (contents NOT available to you;
    filenames only)` -- names, never contents.

If what you would need to judge this item sits behind one of those markers, you
MUST abstain and you MUST NOT score the item low for it. Do not infer quality from
the model's own claim that it succeeded, and do not treat an absent deliverable as
a deliverable that was never produced.

Abstaining is a correct, expected answer and it costs you nothing. An abstention is
recorded and then leaves the comparison entirely -- it is not read as a low score,
not read as a high one, and no rate is computed over it. Say what was missing in
`why_unverifiable`: the turn number you needed, or the deliverable you would have
had to open. A judgment made from a marker instead of evidence is the one answer
this audit has no use for.


List in `evidence_turns` the turn numbers you relied on. Rate only this dimension.

Return JSON matching the schema. Set `dimension` to "Outcome quality".

```

### Response schema

```json
{
  "type": "object",
  "properties": {
    "dimension": {
      "type": "string",
      "enum": [
        "Outcome quality",
        "Communication quality",
        "Trust & grounding",
        "Interaction efficiency",
        "Tool & connector reliability",
        "Memory & personalization"
      ]
    },
    "rating": {
      "type": [
        "integer",
        "null"
      ]
    },
    "not_applicable": {
      "type": "boolean"
    },
    "cannot_determine": {
      "type": "boolean"
    },
    "why_unverifiable": {
      "type": "string"
    },
    "evidence_turns": {
      "type": "array",
      "items": {
        "type": "integer"
      }
    },
    "reasoning": {
      "type": "string"
    },
    "confidence": {
      "type": "string",
      "enum": [
        "low",
        "medium",
        "high"
      ]
    }
  },
  "required": [
    "dimension",
    "rating",
    "not_applicable",
    "cannot_determine",
    "why_unverifiable",
    "evidence_turns",
    "reasoning",
    "confidence"
  ],
  "additionalProperties": false
}
```

## Side-by-side Likert prompt (check 400)

One call per task, carrying both conversations. The budget is split evenly so a long Model A cannot crowd out Model B and bias the comparison toward whichever was rendered first.

```text
## The user's goal

Produce a classroom-ready animation explaining gyroscopic precession.

## Model A

Files accompanying this conversation (contents NOT available to you; filenames only): gyroscope_animation.mp4

NOTE: 3 of 16 model turns delivered a file, image, or video whose content is not present in this transcript. Those turns are labelled below.

[turn 1] USER:
You said I'm a physics teacher, and I want to show my students a short animation explaining how a gyroscope works, specifically why pushing on it makes it tilt sideways instead of the way you'd expect. Can you make me one? It should build up in three steps: First (roughly 0–3s), just the spinning part: a simple circle labeled "FLYWHEEL" with curved arrows around it showing it spinning fast, and a little counter next to it reading "9,750 RPM". Then (3–7s), a square labeled "GIMBAL FRAME" appears around the flywheel, and a blue arrow pushes down on the frame from the left — that's the external force. Finally (7–11s), the frame tilts, and a bright amber arrow labeled "PRECESSION FORCE" shoots out at 90 degrees to the blue one. I want the color coding to be consistent so students immediately see blue = what you push, amber = what actually happens. The whole thing should come out to an 11-second clip, and I'd like it in 4K so it holds up on the classroom projector.

[turn 1] MODEL:  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality cannot be judged from this text]
Your video is ready! [video] 0:00 / 0:10

[turn 2] USER:
You said Thanks for this initial render! Before we make further video tweaks, please follow this instruction for our workflow: From now on, do NOT generate or render a new video on each turn when I request modifications. Instead, acknowledge my requested edits, confirm the keyframe/script updates, and wait until I explicitly say 'Generate' or 'Render' before outputting the next video clip. First, fix Label Errors: In the 7–11s phase, keep the blue downward force arrow labeled as 'PUSH FORCE' (or unlabeled blue force) and ensure ONLY the bright amber vector shooting out at 90 degrees is labeled 'PRECESSION FORCE'. Please confirm these changes and update the internal script, but do not render the video yet."

[turn 2] MODEL:
Understood! I will hold off on rendering any new video until you explicitly give the word "Generate" or "Render." Here is the updated script and keyframe breakdown incorporating your requested label fix: Updated Video Script Breakdown Phase 1 (0–3s): Initial Spin Visuals: Simple circle labeled "FLYWHEEL" spinning rapidly with curved rotation arrows surrounding it. Overlay: Digital counter reading "9,750 RPM" . Phase 2 (3–7s): Applied Force Visuals: A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. Action: A blue vector arrow labeled "PUSH FORCE" pushes down on the left side of the gimbal frame. Phase 3 (7–11s): Gyroscopic Precession Response Visuals: The gimbal frame tilts in response. Vectors & Labels: * The blue downward vector remains labeled "PUSH FORCE" . A bright amber vector shoots out at a 90-degree angle relative to the blue force arrow, clearly and exclusively labeled "PRECESSION FORCE" . Whenever you're ready to see the new render, just say "Generate" or "Render" !

[turn 3] USER:
You said Also, ensure the text 'FLYWHEEL' remains fully visible on the central disc without clipping.

[turn 3] MODEL:
Understood! I have updated the script specifications to ensure that the "FLYWHEEL" label is properly sized, centered, and padded within the central disc so that it remains 100% visible with no clipping throughout the entire clip. Updated Script & Layout Summary Phase 1 (0–3s): Initial Spin Central disc with curved spin arrows and digital counter reading "9,750 RPM" . Text Fix: "FLYWHEEL" label centered cleanly on the central disc with generous margins to prevent clipping. Phase 2 (3–7s): Applied Force "GIMBAL FRAME" appears around the spinning flywheel. Blue vector arrow labeled "PUSH FORCE" pushes down on the left side of the gimbal frame. Phase 3 (7–11s): Precession Response Gimbal frame tilts. Blue downward arrow remains labeled "PUSH FORCE" . Bright amber vector shoots out at 90 degrees relative to the blue force, labeled "PRECESSION FORCE" . Whenever you are ready for me to create the updated video, just say "Generate" or "Render" !

[turn 4] USER:
You said Another point: ensure the blue arrow remains pressing downward on the left side of the gimbal frame throughout 3–11s.

[turn 4] MODEL:
Got it! I have updated the script specifications so that the blue "PUSH FORCE" arrow remains continuously active and pushing downward on the left side of the gimbal frame throughout both Phase 2 and Phase 3 (from 3 to 11 seconds). Comprehensive Script & Keyframe Summary Phase 1 (0–3s): Initial Spin Visuals: Central circle disc spinning rapidly with curved rotation arrows surrounding it. Overlay: Digital counter reading "9,750 RPM" . Text Formatting: "FLYWHEEL" centered cleanly on the central disc with clear margins to avoid any clipping. Phase 2 (3–7s): Applied Force Visuals: A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. Action: A blue vector arrow labeled "PUSH FORCE" appears on the left side of the gimbal frame, pressing downward. Phase 3 (7–11s): Gyroscopic Precession Response Visuals: The gimbal frame tilts in response. Continuous Force: The blue "PUSH FORCE" arrow remains continuously visible and pressing downward on the left side of the frame from 3 to 11 seconds. Resulting Force: A bright amber vector labeled "PRECESSION FORCE" shoots out at a 90-degree angle relative to the blue force arrow. Whenever you are ready for me to render the updated 4K video clip, just reply with "Generate" or "Render" !

[turn 5] USER:
You said Ensure that the amber precession arrow clearly shoots out perpendicular to it at the 7-second mark.

[turn 5] MODEL:
Understood! I have updated the timing and alignment specifications so that precisely at the 7-second mark , the bright amber "PRECESSION FORCE" vector shoots out perpendicular (at a exact 90-degree angle) to the blue downward "PUSH FORCE" vector. Complete Script & Keyframe Summary Resolution & Duration: 4K, 11 seconds total. Phase 1 (0–3s): Initial Spin Central circular disc with curved rotation arrows and a digital counter reading "9,750 RPM" . The text label "FLYWHEEL" is centered on the central disc with clear padding to ensure zero clipping. Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side of the frame, pushing downward. Phase 3 (7–11s): Gyroscopic Precession Response At exactly 7 seconds , the gimbal frame tilts, and a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active and pressing downward on the left side of the frame throughout 3–11s. Color coding is strictly maintained: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Whenever you are ready to render the final video, just reply with "Generate" or "Render" !

[turn 6] USER:
You said Before I tell you to generate, let's adjust the background and lighting. Classroom projectors wash out dark grays. Can you change the background to a clean, light off-white grid pattern with soft Studio lighting so the blue and amber arrows pop from the back of the room?

[turn 6] MODEL:
Understood! I have updated the background and lighting settings in the script to ensure high visibility and contrast on classroom projectors. Comprehensive Script & Visual Summary Format & Resolution: 4K, 11 seconds total. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting, maximizing contrast so the blue and amber vectors stand out clearly. Phase 1 (0–3s): Initial Spin Central circular disc spinning rapidly with curved rotation arrows. Digital counter next to it reading "9,750 RPM" . Text label "FLYWHEEL" centered cleanly on the central disc with generous padding to prevent clipping. Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side of the frame, pressing downward. Phase 3 (7–11s): Gyroscopic Precession Response At exactly the 7-second mark , the gimbal frame tilts. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active and pressing downward on the left side of the frame throughout 3–11s. Color coding is strictly maintained: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Whenever you are ready for me to render the final video, just reply with "Generate" or "Render" !

[turn 7] USER:
You said One quick layout check before we render: what camera angle are we using? Make sure the camera is positioned at a slightly raised 3/4 isometric angle so the 90-degree tilt and spatial depth are completely obvious when the frame rotates.

[turn 7] MODEL:
Understood! I have set the camera perspective to a slightly raised 3/4 isometric angle . This perspective will clearly show the 3D spatial depth and ensure the 90-degree perpendicular relationship between the applied blue force and the resulting amber precession force is visually obvious when the gimbal frame rotates. Final Script & Visual Breakdown Format, Resolution & Camera: 4K, 11 seconds total, slightly raised 3/4 isometric angle. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting for maximum classroom projection contrast. Phase 1 (0–3s): Initial Spin Central circular disc spinning rapidly with curved rotation arrows. Digital counter reading "9,750 RPM" . "FLYWHEEL" label centered cleanly on the central disc with clear padding to prevent clipping. Phase 2 (3–7s): Applied Force Square frame labeled "GIMBAL FRAME" appears around the flywheel. Blue vector arrow labeled "PUSH FORCE" pushes downward on the left side of the frame. Phase 3 (7–11s): Precession Response At exactly 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active and pressing downward on the left side from 3–11s. Consistent color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Whenever you are ready to render the video, just say "Generate" or "Render" !

[turn 8] USER:
You said Almost ready to render! For Phase 1 (0–3s), can you confirm that the '9,750 RPM' counter rapidly counts up from 0 to 9,750 rather than staying static?

[turn 8] MODEL:
Confirmed! I have updated the script so that during Phase 1 (0–3s), the digital counter rapidly counts up from 0 to 9,750 RPM as the flywheel accelerates, rather than remaining static. Final Script & Visual Specifications Format, Resolution & Camera Angle: 4K, 11 seconds total duration, slightly raised 3/4 isometric angle to clearly show 3D depth and frame rotation. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting for high contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows around it. "FLYWHEEL" label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force Square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. Blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response At exactly 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" shoots out perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Everything is fully configured and locked in! Whenever you are ready, just reply with "Generate" or "Render" to produce the video.

[turn 9] USER:
You said Always for Phase 1 (0–3s), can you confirm that the amber arrow at 7s extends smoothly outward rather than snapping in instantly?"

[turn 9] MODEL:
Confirmed! I have updated the script specification so that at the 7-second mark , the bright amber "PRECESSION FORCE" vector extends smoothly and fluidly outward from the gimbal frame over a short fraction of a second, rather than snapping into existence instantaneously. Complete Final Script & Animation Specifications Format, Resolution & Perspective: 4K resolution, 11 seconds total duration, rendered from a slightly raised 3/4 isometric angle to highlight 3D depth and frame tilt clearly. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting to ensure maximum visual contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows around it. "FLYWHEEL" text label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Consistent color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Everything is fully set! Whenever you are ready to produce the video, just reply with "Generate" or "Render" .

[turn 10] USER:
You said Before rendering, can we add a small textual overlay at the bottom during 7–11s that reads: 'Applied force creates torque displaced 90° in the direction of rotation' so students can read the rule while watching?

[turn 10] MODEL:
Confirmed! I have added the textual overlay to the script for Phase 3 (7–11s). It will appear at the bottom of the screen to give students the formal physical rule right as the precession effect occurs. Complete Final Script & Animation Specifications Format, Resolution & Perspective: 4K resolution, 11 seconds total duration, rendered from a slightly raised 3/4 isometric angle to highlight 3D depth and frame tilt clearly. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting to ensure maximum visual contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows around it. "FLYWHEEL" text label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Text Overlay: A clear textual caption appears at the bottom reading: "Applied force creates torque displaced 90° in the direction of rotation" Consistent color coding: Blue = applied force ( "PUSH FORCE" ), Amber = resulting force ( "PRECESSION FORCE" ). Everything is fully updated! Whenever you are ready to produce the video, just reply with "Generate" or "Render" .

[turn 11] USER:
You said Confirm if the last frame (at second 11) smoothly aligns with second 0 so the video loops seamlessly on repeat on my projector.

[turn 11] MODEL:
Confirmed! I have updated the script specification so that at second 11, the visual elements (gimbal frame, force vectors, and text overlay) smoothly reset and fade back to the starting state at second 0, ensuring the clip loops continuously and seamlessly on your projector without any abrupt cuts. Complete Final Script & Animation Specifications Format, Resolution & Loop: 4K resolution, 11 seconds total duration, configured for a seamless visual reset at 11s back to 0s for endless repeating playback. Camera Angle & Perspective: Slightly raised 3/4 isometric angle to clearly highlight 3D spatial depth, gimbal frame tilt, and force vector orientations. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting, optimized for maximum visual contrast on classroom projectors. Phase 1 (0–3s): Initial Spin Up Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" text label centered on the disc with clear margins to prevent clipping. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force A square frame labeled "GIMBAL FRAME" appears around the spinning flywheel. A blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains active and pressing downward on the left side from 3 to 11s. Text Overlay: A clear caption at the bottom reads: "Applied force creates torque displaced 90° in the direction of rotation" Seamless Loop Transition: At 11 seconds, elements smoothly fade/transition back to the 0s initial state for seamless repeating playback. Everything is fully configured! Whenever you are ready to produce the video, just reply with "Generate" or "Render" .

[turn 12] USER:
You said Render the video now! Once generated, please do not make further video updates until I review it and give feedback.

[turn 12] MODEL:  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality cannot be judged from this text]
Your video is ready! [video] 0:00 / 0:10

[turn 13] USER:
You said Let's fix a few errors before final export: keep the '9,750 RPM' counter steady at 9,750 RPM throughout 3–11s instead of dropping to 340 RPM.

[turn 13] MODEL:
Understood! I have updated the script specification so that once the counter reaches 9,750 RPM at the end of Phase 1 (second 3), it remains steady at 9,750 RPM throughout Phases 2 and 3 (3–11s) without dropping. Updated Script & Keyframe Summary Format, Resolution & Loop: 4K resolution, 11 seconds total duration, seamless transition from 11s back to 0s for looping. Camera Angle & Perspective: Slightly raised 3/4 isometric angle to clearly highlight 3D depth, gimbal frame tilt, and vector directions. Environment & Lighting: Clean, light off-white background with a subtle grid pattern and soft studio lighting for clear projection contrast. Phase 1 (0–3s): Spin Up Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" label centered cleanly on the disc without clipping. Counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force "GIMBAL FRAME" appears around the flywheel. Blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. Counter Fix: Digital counter stays steady at 9,750 RPM . Phase 3 (7–11s): Precession Response & Text Overlay At 7s, the gimbal frame tilts in 3D space. Bright amber vector labeled "PRECESSION FORCE" smoothly extends outward perpendicular (90°) to the blue push arrow. Blue "PUSH FORCE" arrow remains continuously active pressing downward on the left side (3–11s). Digital counter remains rock-steady at 9,750 RPM . Bottom Overlay: "Applied force creates torque displaced 90° in the direction of rotation" Whenever you're ready to render the revised video, just reply with "Generate" or "Render" !

[turn 14] USER:
You said Another thing. Let's switch to a dark slate gray or black background. What do you think?

[turn 14] MODEL:
Switching to a dark slate gray background is a great idea, especially if you can dim the classroom lights! Design Considerations: Visual Impact: Dark slate gray creates a sleek, high-contrast environment where glowing or high-saturation vectors—like the bright blue PUSH FORCE and vibrant amber PRECESSION FORCE —will pop dramatically off the screen. Slate Gray vs. Pitch Black: Dark slate gray is actually preferable to pure pitch black because it maintains subtle spatial depth, shadows, and grid lines from the 3/4 isometric perspective without looking flat or harsh. Projector Note: As long as the classroom isn't flooded with bright sunlight, bright white labels and vivid neon force arrows on dark slate gray will be extremely legible from the back of the room. Updated Script & Keyframe Summary Format, Resolution & Loop: 4K resolution, 11 seconds total duration, seamless loop reset at 11s back to 0s. Camera Angle & Perspective: Slightly raised 3/4 isometric angle. Environment & Lighting: Dark slate gray background with subtle grid accents and crisp, high-contrast studio lighting so the flywheel, frame, and vectors glow clearly. Phase 1 (0–3s): Spin Up Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" text centered on the disc with clear padding. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Force "GIMBAL FRAME" appears around the spinning flywheel. Blue vector arrow labeled "PUSH FORCE" appears on the left side, pressing downward. RPM Counter Fix: Counter stays steady at 9,750 RPM throughout this phase. Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7s, the gimbal frame tilts in 3D space. Bright amber vector labeled "PRECESSION FORCE" extends smoothly outward perpendicular (90°) to the blue push arrow. Blue "PUSH FORCE" arrow remains continuously active pressing downward on the left side (3–11s). RPM Counter Fix: Counter remains steady at 9,750 RPM . Bottom Overlay: "Applied force creates torque displaced 90° in the direction of rotation" in clear, bright white text. Whenever you're ready to see the new dark-theme render, just say "Generate" or "Render" !

[turn 15] USER:
You said Last issue. Make sure all text labels stay bright white for high contrast against the dark background.

[turn 15] MODEL:
Understood! I have updated the specifications so that all text labels—including "FLYWHEEL" , "GIMBAL FRAME" , "PUSH FORCE" , "PRECESSION FORCE" , the RPM counter , and the bottom rule overlay —are rendered in crisp, bright white for maximum readability against the dark slate gray background. Complete Final Script & Design Specifications Format, Resolution & Loop: 4K resolution, 11 seconds total duration, designed for a seamless loop transition from 11s back to 0s. Camera Angle & Perspective: Slightly raised 3/4 isometric angle to emphasize 3D spatial depth, gimbal frame tilt, and vector orientations. Environment & Background: Dark slate gray background with subtle grid accents and soft, high-contrast studio lighting. Typography: All text elements and labels rendered in bright white for high contrast and legibility on classroom projectors. Keyframe Sequence Phase 1 (0–3s): Initial Acceleration Central circular flywheel spinning rapidly with curved rotation arrows. "FLYWHEEL" label in bright white, centered cleanly on the disc with clear padding. Digital RPM counter rapidly counts up from 0 to 9,750 RPM . Phase 2 (3–7s): Applied Downward Force Square frame labeled "GIMBAL FRAME" (bright white) appears around the flywheel. Vibrant blue vector arrow labeled "PUSH FORCE" (bright white) appears on the left side, pressing downward. Digital RPM counter holds steady at 9,750 RPM . Phase 3 (7–11s): Precession Response & Rule Overlay At precisely 7 seconds , the gimbal frame tilts in 3D space. Simultaneously at 7s, a bright amber vector labeled "PRECESSION FORCE" (bright white) extends smoothly outward perpendicular (90°) to the blue push arrow. The blue "PUSH FORCE" arrow remains continuously active pressing downward on the left side (3–11s). Digital RPM counter remains rock-steady at 9,750 RPM . Bottom Caption Overlay: "Applied force creates torque displaced 90° in the direction of rotation" in clear, bright white text. Whenever you are ready to produce the new render, just reply with "Generate" or "Render" !

[turn 16] USER:
You said Try to generate it again.

[turn 16] MODEL:  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality cannot be judged from this text]
Your video is ready! [video] 0:00 / 0:10

## Model B

NOTE: 1 of 2 model turns delivered a file, image, or video whose content is not present in this transcript. Those turns are labelled below.

[turn 1] USER:
I'm a physics teacher. Make me a short animation explaining how a gyroscope resists changes in orientation, for high schoolers.

[turn 1] MODEL:  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality cannot be judged from this text]
Your video is ready! 0:00 / 0:10

[turn 2] USER:
Make the precession arrow red instead of blue so it contrasts with the angular momentum vector.

[turn 2] MODEL:
I have updated the internal script specifications. The precession arrow is now red, and the angular momentum vector remains blue along the spin axis. Everything is fully configured and locked in for the next render, including the 4K export setting you asked about.

## Your task

Compare the two conversations and place your preference on a 1-7 scale:

  1 = Model A is much better
  2 = Model A is better
  3 = Model A is slightly better
  4 = the two are equivalent
  5 = Model B is slightly better
  6 = Model B is better
  7 = Model B is much better

Judge which conversation better served the user's goal overall. Set
`cannot_determine` true with `likert` null only if neither transcript contains
enough to compare at all -- a comparison you can make only weakly should still be
a rating, with `confidence` set to low. Truncation on one side alone is enough to
abstain when the difference you would be ranking on sits in the part of that side
you were not shown: the two conversations share one render budget, so each is cut
sooner here than it would be on its own.


Use `cannot_determine` when the transcript does not contain what you would need in
order to judge. The conversation above says where it is incomplete, and these are
the exact markers:

  - `[... conversation truncated here; N exchanges total ...]` -- the conversation
    runs to N exchanges in total and everything past this line was cut. It
    happened; you were not shown it.
  - `[... N characters omitted ...]` -- the middle of that one turn was cut.
  - `[DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its
    quality cannot be judged from this text]` -- a file, image, or video was
    delivered here. You can see that it arrived, not what was in it.
  - `Files accompanying this conversation (contents NOT available to you;
    filenames only)` -- names, never contents.

If what you would need to judge this item sits behind one of those markers, you
MUST abstain and you MUST NOT score the item low for it. Do not infer quality from
the model's own claim that it succeeded, and do not treat an absent deliverable as
a deliverable that was never produced.

Abstaining is a correct, expected answer and it costs you nothing. An abstention is
recorded and then leaves the comparison entirely -- it is not read as a low score,
not read as a high one, and no rate is computed over it. Say what was missing in
`why_unverifiable`: the turn number you needed, or the deliverable you would have
had to open. A judgment made from a marker instead of evidence is the one answer
this audit has no use for.


Return JSON matching the schema.

```

### Response schema

```json
{
  "type": "object",
  "properties": {
    "likert": {
      "type": [
        "integer",
        "null"
      ]
    },
    "cannot_determine": {
      "type": "boolean"
    },
    "why_unverifiable": {
      "type": "string"
    },
    "reasoning": {
      "type": "string"
    },
    "confidence": {
      "type": "string",
      "enum": [
        "low",
        "medium",
        "high"
      ]
    }
  },
  "required": [
    "likert",
    "cannot_determine",
    "why_unverifiable",
    "reasoning",
    "confidence"
  ],
  "additionalProperties": false
}
```

---

# Guards that run before any prompt is sent

**`assert_independent`** (rubric stage) refuses any prompt that binds a contributor L1 label or weight to a criterion ID on the same line. A bare label name is unavoidable because the enum lists all seven; a label attached to a criterion is the leak.

**`assert_blind`** (rating stage) uses the contributor's justification text as a canary. It is long and distinctive, so it cannot appear in a correctly built prompt by chance, and any code path that starts pulling in contributor context will drag it along. It also rejects the framing phrases below, and the contributor's Likert value.

```text
contributor rated
contributor scored
contributor's rating
contributor's score
contributor gave
the contributor believes
existing rating
previous rating
```

Both raise rather than warn. A leak biases every rate toward agreement while still producing plausible-looking numbers, which is the worst possible failure mode for an audit.

---

# What happens to a model's answer

No threshold in this codebase is evaluated by a model. Models emit findings; Python counts them. Validation is strict and one-directional - a dropped finding under-reports defects, which is the safe direction, whereas accepting an invented category would put an unexplainable string in a customer's error-code column. Every drop is counted and reported.

| answer | what happens |
| --- | --- |
| issue category not in the closed set | dropped, counted as `unknown_category` |
| issue with no `evidence` | dropped, counted as `no_evidence` |
| evidence quote not found in the rubric | dropped, counted as `unverifiable_quote` |
| two issues of the same category on one criterion | deduplicated |
| L1 label outside the 7-label enum | dropped, no 200 finding for that criterion |
| weight outside 1-5 | dropped, no 260 finding for that criterion |
| defensible weight range missing or outside 1-5 | dropped, no 260 finding; falling back to the point weight is what over-flagged 260 |
| rating outside the 1-10 scale | error; the dimension leaves the denominator |
| `cannot_determine` | abstention; leaves the denominator, count reported |
| call failed after 3 retries | criterion leaves the denominator, verdict marked low confidence |

## Thresholds the gates apply

All inclusive. Every one of these is a field on `Policy`, not a literal buried in a gate, and `Policy.provenance()` renders the active set so any published number can be reproduced.

| check | fails when |
| --- | --- |
| 200 L1 labels | >= 30% of labelled criteria wrong |
| 230 rubric quality | >= 10% carry major issues |
| 240 rubric quality | >= 15% carry major or moderate |
| 250 rubric quality | >= 20% carry any issue |
| 260 weights | of criteria whose weight is outside the auditor's defensible range, >= 5% two levels outside, or >= 30% any levels outside |
| 270 rubric ratings | >= 5 criteria rated differently, or >= 20% of criteria |
| 300 dimension ratings | >= 2 deltas of 3+, or >= 5 total |
| 400 Likert | delta >= 3 points |
| 280 / 310 turns | >= 3 incorrectly cited turns |

Bands map to scores: fail = 1, non-fail = 3, clean = 5. Score 2 is unused and raises if emitted. Below 50% of judgments surviving abstention, 270 and 300 return `not_evaluated` instead of a rate.

## The 21 dimensions and where each is answered

| id | dimension | sub-dimension | shape | how it is answered |
| --- | --- | --- | --- | --- |
| 70 | Prompt | Domain Relevance | C | not implemented yet |
| 75 | Prompt | Pre-seeded/Opening Prompt Consistency | C | not implemented yet |
| 80 | Target Outcome | - | B | informed stage |
| 85 | Prompt | Environment Context | A | not implemented yet |
| 90 | User Input | Valid Links | A | deterministic |
| 95 | Conversation | Output Artifacts Preserved | A | informed stage |
| 96 | Conversation | Minimum Turns | A | deterministic |
| 100 | Key Turn | Key Turn Identification | C | deterministic |
| 110 | Key Turn | Key Turn Justification | B | informed stage |
| 200 | Rubric Criteria | L1 Labels/Annotations | C | rubric stage |
| 210 | Rubric Criteria | L2 Labels/Annotations | B | blocked on the L2 taxonomy |
| 220 | Rubric | Rubric Autofail | A | informed stage (second opinion) |
| 230 | Rubric Criteria | Overall Rubric Quality - 10% | C | rubric stage |
| 240 | Rubric Criteria | Overall Rubric Quality - 15% | C | rubric stage |
| 250 | Rubric Criteria | Overall Rubric Quality - 20% | C | rubric stage |
| 260 | Rubric Criteria | Weights | C | rubric stage |
| 270 | Rating | Rubric Evaluation | C | rating stage (blind) |
| 280 | Rubric Ratings | Relevant Turns | C | informed stage |
| 300 | Rating | Correctness/Accuracy - Major Rating | C | rating stage (blind) |
| 310 | Dimension Ratings | Relevant Turns | C | informed stage |
| 400 | SxS Rating | Ranking Disagreement | C | rating stage (blind) |
| 450 | Dimension Rating / Model Ranking | Justifications | C | informed stage |
| 460 | SxS Rating | Inconsistent Ranking | A | deterministic |
| 470 | SxS Rating | Verdict | A | informed stage |
| 1000 | Unauditable | - | A | not implemented yet |

Shape A is binary (fail or clean). Shape B can never fail a task, only report non-fail. Shape C is graded. The shape is enforced in `build_verdict`, so a Shape B check cannot emit a fail even if a model asks for one.

